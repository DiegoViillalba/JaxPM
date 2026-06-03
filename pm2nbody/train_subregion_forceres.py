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

def eval_region(model, params, feats, vel, a, target, region_idx, prefix):
    """Compute metrics for a given subset of particles."""
    feats_r  = feats[region_idx]
    vel_r    = vel[region_idx]
    target_r = target[region_idx]
    pred_r   = jax.device_get(model.apply(params, feats_r, vel_r, jnp.array(a)))
    pred_r   = np.asarray(pred_r)
    tgt_r    = np.asarray(jax.device_get(target_r))

    mse     = float(np.mean((pred_r - tgt_r) ** 2))
    r_vals  = [pearsonr(pred_r[:, c], tgt_r[:, c])[0] for c in range(3)]
    r_mean  = float(np.mean(r_vals))
    frac_sq = float(np.mean(
        np.sum((pred_r - tgt_r)**2, axis=1) / (np.sum(tgt_r**2, axis=1) + 1e-12)
    ))
    return {
        f"{prefix}mse":      mse,
        f"{prefix}pearson_r": r_mean,
        f"{prefix}frac_mse": frac_sq,
    }


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
    logger.info("Computing force pair (coarse→fine, same particles) …")
    _force_pair_jit = jax.jit(partial(compute_force_pair, mesh_lr=mesh_lr, mesh_hr=mesh_hr))
    f_coarse, f_fine, delta_f = _force_pair_jit(pos_tr)
    df_mag = float(jnp.mean(jnp.sqrt(jnp.sum(delta_f**2, axis=-1))))
    f_mag  = float(jnp.mean(jnp.sqrt(jnp.sum(f_fine**2,  axis=-1))))
    logger.info(f"  |F_fine| mean = {f_mag:.4e}   |ΔF| mean = {df_mag:.4e}  "
                f"({df_mag/f_mag:.1%} of F_fine)")

    # ── Sample weights (train region only) ────────────────────────────────────
    weights_all = compute_sample_weights(
        pos_tr, det_D_all, n_part,
        sc_boost=loss_sc_boost,
        density_boost=loss_dens_boost,
        density_gamma=loss_dens_gamma,
    )

    # ── Slice to training region ──────────────────────────────────────────────
    feats_tr   = feats_all[train_idx]
    vel_feat_tr = vel_tr[train_idx] if use_velocity else jnp.zeros((len(train_idx), 3))
    target_tr  = delta_f[train_idx]
    weights_tr = weights_all[train_idx]

    logger.info(f"Train region  |ΔF| mean = "
                f"{float(jnp.mean(jnp.sqrt(jnp.sum(target_tr**2, axis=-1)))):.4e}")
    logger.info(f"Test  region  |ΔF| mean = "
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

            # ── Train region metrics ───────────────────────────────────────
            tm = eval_region(lag_model, params, feats_all, vel_tr if use_velocity
                             else jnp.zeros_like(vel_tr), a_tr, delta_f,
                             train_idx, "train/")
            log.update(tm)

            # ── Test region metrics (generalisation) ──────────────────────
            vm = eval_region(lag_model, params, feats_all, vel_tr if use_velocity
                             else jnp.zeros_like(vel_tr), a_tr, delta_f,
                             test_idx, "test/")
            log.update(vm)

            wandb.log(log, step=step)
            logger.info(
                f"step {step:5d} | loss={float(loss):.3e} | "
                f"train R={tm['train/pearson_r']:.3f}  "
                f"test R={vm['test/pearson_r']:.3f}  "
                f"(gap={tm['train/pearson_r'] - vm['test/pearson_r']:+.3f})"
            )

            if vm["test/mse"] < best_test_mse:
                best_test_mse = vm["test/mse"]
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
