"""
train_subregion_forceres.py — Sub-region force-resolution correction.

Concept
-------
Within a SINGLE simulation:
  - Coarse force:  PM on mesh_lr  (standard simulation force)
  - Fine force:    PM on mesh_hr  (more accurate, same particles)
  - ΔF = F_fine − F_coarse  (target, force-resolution correction)

Training set: particles in a spatial sub-region (Lagrangian patch).
Test set:     the rest of the simulation.

Goal: does the model learn from a sub-region and generalise to the full sim?
Both train loss and test loss are logged at every step.

Data:
  Output of generate_data_single.py (or generate_data_disp.py LR files):
    pos_m{n_part}_s{sim_id}.npy  [n_snaps, n_part³, 3]  positions (Mpc/h)
    vel_m{n_part}_s{sim_id}.npy  [n_snaps, n_part³, 3]  velocities
    scale_factors.npy            [n_snaps]

Usage:
  python train_subregion_forceres.py --config configs/subregion_forceres.yaml
"""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.85"

import yaml
import pickle
import logging
import shutil
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import optax
import wandb
from scipy.stats import pearsonr

from jaxpm.lagrangian import (
    get_axis_neighbor_indices,
    get_shell_neighbor_indices,
    make_lagrangian_corrector,
)

# ── Reuse from existing pipelines ─────────────────────────────────────────────
from train_lag_force import (
    compute_force_pair,     # same particles, two mesh sizes → (f_lr, f_hr, ΔF)
    snapshot_features,      # Lagrangian features
    make_train_step,        # JIT-compiled MLP train step
    compute_sample_weights, # per-particle loss weights
    compute_metrics,        # Pearson R, frac_mse, etc.
)
# ── Optional CNN stage-1 prior ─────────────────────────────────────────────────
from train_lag_massres import (
    _load_cnn_massres_checkpoint,       # loads checkpoint saved by train_cnn_forceres.py
    compute_cnn_massres_correction,     # ∇ΔΦ_CNN evaluated at positions (mesh_lr units)
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")


# ==============================================================================
# 1. Data loading (single simulation)
# ==============================================================================

def load_single_snapshot(
    data_dir: Path,
    sim_id: int,
    snap_idx: int,
    n_part: int,
    box_size: float,
) -> tuple:
    """
    Load one snapshot of a single-simulation dataset.

    Returns (pos, vel, a) where:
      pos  [n_part³, 3]  in mesh (n_part) units: pos_mpc * n_part / box_size
      vel  [n_part³, 3]  same units
      a    float          scale factor
    """
    data_dir = Path(data_dir)
    scale    = float(n_part) / float(box_size)   # Mpc/h → n_part mesh units
    avals    = np.load(data_dir / "scale_factors.npy")
    a        = float(avals[snap_idx])

    pos_mpc  = np.load(data_dir / f"pos_m{n_part}_s{sim_id}.npy")[snap_idx]
    vel_mpc  = np.load(data_dir / f"vel_m{n_part}_s{sim_id}.npy")[snap_idx]

    return (
        jnp.array(pos_mpc * scale, dtype=jnp.float32),
        jnp.array(vel_mpc * scale, dtype=jnp.float32),
        a,
    )


# ==============================================================================
# 2. Spatial split: sub-region (train) vs rest (test)
# ==============================================================================

def make_patch_split(
    n_part: int,
    train_patch_n: int,
    seed: int = 0,
) -> tuple:
    """
    Split particles into a training sub-region and the rest.

    Training: a contiguous Lagrangian cube of size train_patch_n³
              (wraps periodically at box boundary)
    Test:     all remaining particles

    Returns
    -------
    train_idx : [train_patch_n³]              flat particle indices (training)
    test_idx  : [n_part³ - train_patch_n³]    flat particle indices (test)
    patch_n   : int  actual patch side length used
    """
    rng = np.random.default_rng(seed)
    ox, oy, oz = rng.integers(0, n_part, size=3)

    ix = (np.arange(train_patch_n) + ox) % n_part
    iy = (np.arange(train_patch_n) + oy) % n_part
    iz = (np.arange(train_patch_n) + oz) % n_part

    IXX, IYY, IZZ = np.meshgrid(ix, iy, iz, indexing='ij')
    train_idx = (IXX.ravel() * n_part**2
                 + IYY.ravel() * n_part
                 + IZZ.ravel()).astype(np.int32)

    all_mask = np.ones(n_part**3, dtype=bool)
    all_mask[train_idx] = False
    test_idx = np.where(all_mask)[0].astype(np.int32)

    return train_idx, test_idx, train_patch_n


# ==============================================================================
# 3. Validation
# ==============================================================================

def eval_region(
    model, params, feats, vel, a, delta_f_full,
    region_idx, prefix,
    cnn_pred_all=None,
):
    """
    Compute metrics for a given subset of particles.

    Parameters
    ----------
    delta_f_full : [N, 3]   full ΔF target (F_fine − F_coarse) for ALL particles
    cnn_pred_all : [N, 3]   CNN stage-1 prediction for ALL particles (or None)
                            When provided, the MLP is evaluated against the residual
                            target, and total (CNN + MLP) metrics are also logged.
    """
    feats_r  = feats[region_idx]
    vel_r    = vel[region_idx]
    delta_f_r = np.asarray(jax.device_get(delta_f_full[region_idx]))

    # MLP target: residual if two-stage, full ΔF otherwise
    if cnn_pred_all is not None:
        cnn_r = np.asarray(cnn_pred_all[region_idx])
        target_r = delta_f_r - cnn_r
    else:
        cnn_r = np.zeros_like(delta_f_r)
        target_r = delta_f_r

    mlp_pred_r = np.asarray(jax.device_get(
        model.apply(params, feats_r, vel_r, jnp.array(a))
    ))
    total_pred_r = cnn_r + mlp_pred_r       # CNN + MLP combined

    def _metrics(pred, tgt, tag):
        mse    = float(np.mean((pred - tgt) ** 2))
        r_mean = float(np.mean([pearsonr(pred[:, c], tgt[:, c])[0] for c in range(3)]))
        fmse   = float(np.mean(
            np.sum((pred - tgt)**2, axis=1) / (np.sum(tgt**2, axis=1) + 1e-12)
        ))
        return {f"{tag}mse": mse, f"{tag}pearson_r": r_mean, f"{tag}frac_mse": fmse}

    out = {}
    # MLP-only metrics (vs residual target)
    out.update(_metrics(mlp_pred_r, target_r, prefix))
    # Total correction vs full ΔF (the physically meaningful metric)
    out.update(_metrics(total_pred_r, delta_f_r, f"{prefix}total_"))
    # CNN-only metrics (fixed, not training)
    if cnn_pred_all is not None:
        out.update(_metrics(cnn_r, delta_f_r, f"{prefix}cnn_"))
    return out


# ==============================================================================
# 4. Main training loop
# ==============================================================================

def train(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    exp_cfg   = SimpleNamespace(**cfg.get("experiment", {}))
    data_cfg  = SimpleNamespace(**cfg["data"])
    model_cfg = SimpleNamespace(**cfg["model"])
    train_cfg = SimpleNamespace(**cfg["training"])
    wb_cfg    = cfg.get("wandb", {})

    wandb.init(
        project = wb_cfg.get("project", "pm2nbody_subregion_forceres"),
        name    = getattr(exp_cfg, "name", None),
        tags    = wb_cfg.get("tags", []),
        config  = cfg,
    )
    out_dir = Path(getattr(exp_cfg, "output_dir", "runs/subregion_forceres")) / wandb.run.name
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, out_dir / "config.yaml")

    # ── Config ────────────────────────────────────────────────────────────────
    data_dir    = Path(data_cfg.data_dir)
    n_part      = int(data_cfg.n_part)
    mesh_lr     = int(data_cfg.mesh_lr)      # coarse force mesh
    mesh_hr     = int(data_cfg.mesh_hr)      # fine force mesh
    box_size    = float(data_cfg.box_size)
    snap_train  = int(data_cfg.snap_train)
    sim_train   = int(getattr(data_cfg, "sim_id_train", 0))
    sim_val     = int(getattr(data_cfg, "sim_id_val",   1))
    snaps_val   = list(getattr(data_cfg, "snaps_val",   [snap_train]))

    use_strain     = bool(getattr(model_cfg, "use_strain",     True))
    use_invariants = bool(getattr(model_cfg, "use_invariants", False))
    use_velocity   = bool(getattr(model_cfg, "use_velocity",   True))
    n_shell        = int(getattr(model_cfg,  "n_shell",        0))
    env_pool_mode  = str(getattr(model_cfg,  "env_pool_mode",  "mean_var"))
    hidden_dim     = int(getattr(model_cfg,  "hidden_dim",     64))
    n_layers       = int(getattr(model_cfg,  "n_layers",       3))
    cnn_ckpt_path  = getattr(model_cfg, "cnn_checkpoint", None)

    n_steps      = int(getattr(train_cfg,   "n_steps",       2000))
    lr_val       = float(getattr(train_cfg,  "lr",            3e-4))
    weight_decay = float(getattr(train_cfg,  "weight_decay",  1e-4))
    warmup       = int(getattr(train_cfg,    "warmup_steps",    50))
    log_every    = int(getattr(train_cfg,    "log_every",       50))
    save_every   = int(getattr(train_cfg,    "save_every",     200))
    seed         = int(getattr(train_cfg,    "seed",             0))

    # Sub-region config
    train_patch_n   = int(getattr(train_cfg, "train_patch_n",   n_part // 2))
    patch_seed      = int(getattr(train_cfg, "patch_seed",      42))

    loss_sc_boost   = float(getattr(train_cfg, "loss_sc_boost",      1.0))
    loss_dens_boost = float(getattr(train_cfg, "loss_density_boost", 0.0))
    loss_dens_gamma = float(getattr(train_cfg, "loss_density_gamma", 0.5))

    train_frac = (train_patch_n / n_part) ** 3
    logger.info(f"n_part={n_part}  mesh_lr={mesh_lr}  mesh_hr={mesh_hr}")
    logger.info(f"train_patch_n={train_patch_n}  "
                f"({train_patch_n**3:,} train / {n_part**3 - train_patch_n**3:,} test  "
                f"= {train_frac:.1%} of simulation)")

    # ── Lagrangian neighbour indices ──────────────────────────────────────────
    neighbor_idx = get_axis_neighbor_indices(n_part)
    ext_neighbor_idx, ext_offsets, shell_slices = None, None, ()
    if n_shell > 0:
        ext_neighbor_idx, ext_offsets, shell_slices = \
            get_shell_neighbor_indices(n_part, n_shell)

    # ── Train/test split ──────────────────────────────────────────────────────
    logger.info(f"Creating spatial split (seed={patch_seed}) …")
    train_idx, test_idx, _ = make_patch_split(n_part, train_patch_n, seed=patch_seed)
    logger.info(f"  train region: {len(train_idx):,} particles  |  "
                f"test region: {len(test_idx):,} particles")

    # ── Load training snapshot ────────────────────────────────────────────────
    logger.info(f"Loading sim={sim_train}  snap={snap_train}")
    pos_tr, vel_tr, a_tr = load_single_snapshot(
        data_dir, sim_train, snap_train, n_part, box_size
    )
    logger.info(f"  a_train={a_tr:.4f}  N={pos_tr.shape[0]:,}")

    # ── Lagrangian features (all particles — needed for test evaluation) ──────
    # pos_tr is in n_part units; snapshot_features uses mesh_size=n_part → consistent
    logger.info("Computing Lagrangian features …")
    feats_all, det_D_all = snapshot_features(
        pos_tr, neighbor_idx, n_part,
        use_strain, use_invariants,
        ext_neighbor_idx, ext_offsets, env_pool_mode, shell_slices,
    )
    feat_dim = int(feats_all.shape[1])
    sc_frac  = float(np.mean(jax.device_get(det_D_all) < 0))
    logger.info(f"  feat_dim={feat_dim}  SC={sc_frac:.2%}")

    # ── Force pair (coarse + fine, same particles) ────────────────────────────
    # UNIT FIX: compute_force_pair expects positions in mesh_lr units.
    # load_single_snapshot returns pos in n_part units.
    # Convert: pos_lr = pos * (mesh_lr / n_part)
    scale_to_lr = float(mesh_lr) / float(n_part)
    pos_tr_lr   = pos_tr * scale_to_lr    # [N, 3]  in mesh_lr units

    logger.info(f"Computing force pair  mesh_lr={mesh_lr} vs mesh_hr={mesh_hr} …")
    logger.info(f"  pos unit conversion: n_part={n_part} → mesh_lr={mesh_lr}  (×{scale_to_lr:.3f})")
    _force_pair_jit = jax.jit(partial(compute_force_pair, mesh_lr=mesh_lr, mesh_hr=mesh_hr))
    f_coarse, f_fine, delta_f = _force_pair_jit(pos_tr_lr)
    df_mag = float(jnp.mean(jnp.sqrt(jnp.sum(delta_f**2, axis=-1))))
    f_mag  = float(jnp.mean(jnp.sqrt(jnp.sum(f_fine**2,  axis=-1))))
    logger.info(f"  |F_fine| mean = {f_mag:.4e}   |ΔF| mean = {df_mag:.4e}  "
                f"({df_mag/f_mag:.1%} of F_fine)")

    # ── Optional CNN prior (stage 1) ──────────────────────────────────────────
    # If cnn_checkpoint is set, the CNN predicts ΔF_CNN from the LR density grid.
    # The MLP then learns only the residual: ΔF_mlp = ΔF − ΔF_CNN
    # Total correction: ΔF_total = ΔF_CNN + ΔF_MLP
    cnn_model, cnn_params = None, None
    IS_TWO_STAGE = cnn_ckpt_path is not None
    cnn_pred_all_np = np.zeros_like(np.asarray(delta_f))   # zero placeholder

    if IS_TWO_STAGE:
        logger.info(f"Loading CNN checkpoint: {cnn_ckpt_path}")
        cnn_model, cnn_params = _load_cnn_massres_checkpoint(cnn_ckpt_path)
        n_cnn = sum(x.size for x in jax.tree_util.tree_leaves(cnn_params))
        logger.info(f"  CNN loaded  ({n_cnn:,} params)")

        # CNN correction requires positions in mesh_lr units (same as force pair)
        vel_feat_all = vel_tr if use_velocity else jnp.zeros_like(vel_tr)
        # Scale velocity to mesh_lr units too
        vel_lr_all = vel_feat_all * scale_to_lr

        _cnn_jit = jax.jit(
            lambda pos, vel, a: compute_cnn_massres_correction(
                cnn_model, cnn_params, pos, vel, a, mesh_lr
            )
        )
        cnn_pred_all_np = np.asarray(jax.device_get(
            _cnn_jit(pos_tr_lr, vel_lr_all, jnp.array(a_tr))
        ))

        cnn_r_vals = [pearsonr(cnn_pred_all_np[:, c],
                               np.asarray(delta_f[:, c]))[0] for c in range(3)]
        cnn_r      = float(np.mean(cnn_r_vals))
        residual_mag = float(np.mean(np.sqrt(np.sum(
            (np.asarray(delta_f) - cnn_pred_all_np)**2, axis=-1
        ))))
        logger.info(f"  CNN R_mean={cnn_r:.4f}  |ΔF_CNN|={np.mean(np.sqrt(np.sum(cnn_pred_all_np**2,axis=-1))):.4e}")
        logger.info(f"  Residual |ΔF − ΔF_CNN|={residual_mag:.4e}  "
                    f"(was {df_mag:.4e} → {residual_mag/df_mag:.1%} remaining)")
    else:
        logger.info("Single-stage: MLP trains on full ΔF  (no CNN checkpoint)")

    # ── Sample weights (train region only) ────────────────────────────────────
    # NOTE: compute_sample_weights expects positions in mesh_lr units → use pos_tr_lr
    weights_all = compute_sample_weights(
        pos_tr_lr, det_D_all, mesh_lr,
        sc_boost=loss_sc_boost,
        density_boost=loss_dens_boost,
        density_gamma=loss_dens_gamma,
    )

    # ── MLP target: residual if two-stage, full ΔF otherwise ─────────────────
    delta_f_mlp_target = (
        delta_f - jnp.array(cnn_pred_all_np)
        if IS_TWO_STAGE else delta_f
    )
    stage_label = "F_corrected = F_lr + ΔF_CNN + ΔF_MLP" if IS_TWO_STAGE else "F_corrected = F_lr + ΔF_MLP"
    logger.info(f"Pipeline: {stage_label}")

    # ── Slice to training region ──────────────────────────────────────────────
    feats_tr    = feats_all[train_idx]
    vel_feat_tr = vel_tr[train_idx] if use_velocity else jnp.zeros((len(train_idx), 3))
    target_tr   = delta_f_mlp_target[train_idx]
    weights_tr  = weights_all[train_idx]

    logger.info(f"Train region  MLP target |ΔF_mlp| mean = "
                f"{float(jnp.mean(jnp.sqrt(jnp.sum(target_tr**2, axis=-1)))):.4e}")
    logger.info(f"Test  region  full  |ΔF|  mean = "
                f"{float(jnp.mean(jnp.sqrt(jnp.sum(delta_f[test_idx]**2, axis=-1)))):.4e}")

    # ── Model ─────────────────────────────────────────────────────────────────
    lag_model = make_lagrangian_corrector(
        hidden_dim=hidden_dim, n_layers=n_layers, output_dim=3
    )
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=lr_val, warmup_steps=warmup, decay_steps=n_steps
    )
    optimizer = optax.adamw(schedule, weight_decay=weight_decay)

    rng    = jax.random.PRNGKey(seed)
    params = lag_model.init(rng, feats_tr[:4], vel_feat_tr[:4], jnp.array(a_tr))
    opt_state = optimizer.init(params)
    wandb.log({"model/is_two_stage": int(IS_TWO_STAGE)}, step=0)
    if IS_TWO_STAGE:
        wandb.log({"cnn/pearson_r_train": float(np.mean(
            [pearsonr(cnn_pred_all_np[train_idx, c],
                      np.asarray(delta_f[train_idx, c]))[0] for c in range(3)]
        ))}, step=0)

    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    logger.info(f"MLP: {n_params:,} params  feat_dim={feat_dim}  "
                f"hidden={hidden_dim}×{n_layers}")

    train_step_fn = make_train_step(lag_model, optimizer)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_test_mse = np.inf
    best_params   = None

    for step in range(n_steps):
        params, opt_state, loss = train_step_fn(
            params, opt_state,
            feats_tr, vel_feat_tr, jnp.array(a_tr), target_tr, weights_tr,
        )

        if step % log_every == 0 or step == n_steps - 1:
            log = {"step": step, "train/loss": float(loss)}

            vel_eval = vel_tr if use_velocity else jnp.zeros_like(vel_tr)

            # ── Train region metrics ───────────────────────────────────────
            tm = eval_region(lag_model, params, feats_all, vel_eval, a_tr,
                             delta_f, train_idx, "train/",
                             cnn_pred_all=cnn_pred_all_np if IS_TWO_STAGE else None)
            log.update(tm)

            # ── Test region metrics (generalisation key metric) ────────────
            vm = eval_region(lag_model, params, feats_all, vel_eval, a_tr,
                             delta_f, test_idx, "test/",
                             cnn_pred_all=cnn_pred_all_np if IS_TWO_STAGE else None)
            log.update(vm)

            # Key generalisation metrics for WandB dashboard
            # Use total_ (CNN+MLP vs full ΔF) if two-stage, else MLP vs ΔF
            r_key = "total_pearson_r" if IS_TWO_STAGE else "pearson_r"
            train_r = tm[f"train/{r_key}"]
            test_r  = vm[f"test/{r_key}"]
            log["generalisation/train_R"] = train_r
            log["generalisation/test_R"]  = test_r
            log["generalisation/gap"]     = train_r - test_r

            wandb.log(log, step=step)
            logger.info(
                f"step {step:5d} | loss={float(loss):.3e} | "
                f"train R={train_r:.3f}  "
                f"test R={test_r:.3f}  "
                f"(gap={train_r - test_r:+.3f})"
            )

            # Save best by test-region total MSE
            mse_key = "total_mse" if IS_TWO_STAGE else "mse"
            if vm[f"test/{mse_key}"] < best_test_mse:
                best_test_mse = vm[f"test/{mse_key}"]
                best_params   = hk.data_structures.to_mutable_dict(params)
                with open(out_dir / "best_params.pkl", "wb") as fh:
                    pickle.dump(best_params, fh)

        if step % save_every == 0:
            with open(out_dir / "checkpoint.pkl", "wb") as fh:
                pickle.dump({
                    "params":       hk.data_structures.to_mutable_dict(params),
                    "model_cfg":    vars(model_cfg),
                    "data_cfg":     vars(data_cfg),
                    "train_idx":    train_idx,
                    "test_idx":     test_idx,
                    "step":         step,
                }, fh)

    # ── Final save ────────────────────────────────────────────────────────────
    with open(out_dir / "final_params.pkl", "wb") as fh:
        pickle.dump(hk.data_structures.to_mutable_dict(params), fh)
    with open(out_dir / "checkpoint.pkl", "wb") as fh:
        pickle.dump({
            "params":    hk.data_structures.to_mutable_dict(params),
            "model_cfg": vars(model_cfg),
            "data_cfg":  vars(data_cfg),
            "train_idx": train_idx,
            "test_idx":  test_idx,
            "step":      n_steps - 1,
        }, fh)

    logger.info(f"Training done → {out_dir}")
    logger.info(f"Best test MSE: {best_test_mse:.4e}")
    wandb.finish()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train(args.config)
