"""
train_pretrain_forceres.py — Multi-simulation pre-training for force-res MLP.

Concept
-------
Pre-train the Lagrangian MLP corrector on ALL particles from multiple
simulations and snapshots.  No spatial patch split during pre-training —
every particle contributes equally.

This produces a generic prior that already captures the physics of
ΔF = F_fine − F_coarse across diverse cosmological environments.
The checkpoint can then be fine-tuned cheaply on a small spatial patch
during a live simulation (see finetune_subregion_forceres.py).

Why multi-sim?
--------------
The region-size scan shows that test R keeps rising as the training set
grows.  The bottleneck is not receptive field but diversity of local
environments seen during training.  Training on N_sims × N_snaps full
simulations achieves the same diversity as an 80% single-sim patch, at
the same on-the-fly cost (12.5% fine-tuning patch remains unchanged).

Training loop
-------------
All (sim, snap) datasets are precomputed at startup (features + force
pair + optional CNN correction).  Each training step samples one dataset
uniformly at random — equivalent to a multi-task learning setup where
each task is one snapshot.

Pipeline
--------
  Step 1 (optional):  train_cnn_forceres.py  --config configs/cnn_forceres.yaml
  Step 2 (this):      train_pretrain_forceres.py --config configs/pretrain_forceres.yaml
  Step 3 (on-the-fly): finetune_subregion_forceres.py  (not yet implemented)

Usage
-----
  python train_pretrain_forceres.py --config configs/pretrain_forceres.yaml
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

from train_lag_force import (
    compute_force_pair,
    snapshot_features,
    make_train_step,
    compute_sample_weights,
)
from train_subregion_forceres import load_single_snapshot
from train_lag_massres import (
    _load_cnn_massres_checkpoint,
    compute_cnn_massres_correction,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Dataset builder
# ══════════════════════════════════════════════════════════════════════════════

def build_dataset(
    data_dir, sim_id, snap_idx,
    n_part, mesh_lr, mesh_hr, box_size,
    neighbor_idx,
    use_strain, use_invariants, use_velocity,
    ext_neighbor_idx, ext_offsets, env_pool_mode, shell_slices,
    cnn_model, cnn_params,          # None → single-stage
    sc_boost, density_boost, density_gamma,
):
    """
    Build one (features, target, vel, a, weights) dataset for a single snapshot.

    Positions are converted from n_part units to mesh_lr units before the
    force-pair computation (unit fix).  Features are computed in n_part units
    (consistent with snapshot_features / mesh_size = n_part).

    Returns a dict of numpy / JAX arrays ready for the training step.
    """
    scale_to_lr = float(mesh_lr) / float(n_part)

    pos, vel, a = load_single_snapshot(data_dir, sim_id, snap_idx, n_part, box_size)
    pos_lr = pos * scale_to_lr                         # mesh_lr units for force pair

    # Lagrangian features (positions in n_part units, mesh_size = n_part)
    feats, det_D = snapshot_features(
        pos, neighbor_idx, n_part,
        use_strain, use_invariants,
        ext_neighbor_idx, ext_offsets, env_pool_mode, shell_slices,
    )

    # Force pair
    _fp = jax.jit(partial(compute_force_pair, mesh_lr=mesh_lr, mesh_hr=mesh_hr))
    _, _, delta_f = _fp(pos_lr)

    # Optional CNN stage-1 correction
    if cnn_model is not None:
        vel_lr = vel * scale_to_lr
        _cnn  = jax.jit(lambda p, v, a_: compute_cnn_massres_correction(
            cnn_model, cnn_params, p, v, a_, mesh_lr
        ))
        cnn_pred = np.asarray(jax.device_get(_cnn(pos_lr, vel_lr, jnp.array(a))))
        target   = delta_f - jnp.array(cnn_pred)
    else:
        cnn_pred = None
        target   = delta_f

    det_D_np = np.asarray(jax.device_get(det_D))
    weights  = compute_sample_weights(
        pos_lr, det_D_np, mesh_lr,
        sc_boost=sc_boost, density_boost=density_boost, density_gamma=density_gamma,
    )

    df_mag = float(jnp.mean(jnp.sqrt(jnp.sum(delta_f**2, axis=-1))))
    sc_frac = float(np.mean(det_D_np < 0))

    vel_feat = vel if use_velocity else jnp.zeros_like(vel)

    return {
        "feats":    feats,
        "vel":      vel_feat,
        "target":   target,
        "weights":  weights,
        "a":        a,
        "delta_f":  delta_f,    # full ΔF (for total-R metric)
        "cnn_pred": cnn_pred,   # None if single-stage
        "df_mag":   df_mag,
        "sc_frac":  sc_frac,
        "sim_id":   sim_id,
        "snap_idx": snap_idx,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Validation
# ══════════════════════════════════════════════════════════════════════════════

def validate(model, params, val_datasets, prefix="val/"):
    """
    Evaluate model on all validation datasets.
    Returns mean Pearson R and MSE improvement across snapshots.
    """
    r_vals, mse_improv_vals = [], []
    log = {}

    for ds in val_datasets:
        pred = np.asarray(jax.device_get(
            jax.jit(model.apply)(params, ds["feats"], ds["vel"], jnp.array(ds["a"]))
        ))
        # Total correction: CNN + MLP (or just MLP)
        cnn = ds["cnn_pred"] if ds["cnn_pred"] is not None else np.zeros_like(pred)
        total = cnn + pred
        df    = np.asarray(jax.device_get(ds["delta_f"]))

        r_mean = float(np.mean([pearsonr(total[:, c], df[:, c])[0] for c in range(3)]))
        mse_b  = float(np.mean(df**2))
        mse_c  = float(np.mean((total - df)**2))
        improv = 1 - mse_c / mse_b

        tag = f"s{ds['sim_id']}_snap{ds['snap_idx']}"
        log[f"{prefix}{tag}/pearson_r"]     = r_mean
        log[f"{prefix}{tag}/mse_improv%"]   = improv * 100

        r_vals.append(r_mean)
        mse_improv_vals.append(improv)

    log[f"{prefix}pearson_r_mean"]   = float(np.mean(r_vals))
    log[f"{prefix}mse_improv%_mean"] = float(np.mean(mse_improv_vals)) * 100
    return log, float(np.mean(r_vals))


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Main
# ══════════════════════════════════════════════════════════════════════════════

def train(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    exp_cfg   = SimpleNamespace(**cfg.get("experiment", {}))
    data_cfg  = SimpleNamespace(**cfg["data"])
    model_cfg = SimpleNamespace(**cfg["model"])
    train_cfg = SimpleNamespace(**cfg["training"])
    wb_cfg    = cfg.get("wandb", {})

    wandb.init(
        project = wb_cfg.get("project", "pm2nbody_pretrain_forceres"),
        name    = getattr(exp_cfg, "name", None),
        tags    = wb_cfg.get("tags", []),
        config  = cfg,
    )
    out_dir = Path(getattr(exp_cfg, "output_dir", "runs/pretrain_forceres")) / wandb.run.name
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, out_dir / "config.yaml")

    # ── Config ────────────────────────────────────────────────────────────────
    data_dir        = Path(data_cfg.data_dir)
    n_part          = int(data_cfg.n_part)
    mesh_lr         = int(data_cfg.mesh_lr)
    mesh_hr         = int(data_cfg.mesh_hr)
    box_size        = float(data_cfg.box_size)
    sim_ids_train   = list(data_cfg.sim_ids_train)
    sim_ids_val     = list(getattr(data_cfg, "sim_ids_val",   []))
    snaps_train     = list(data_cfg.snaps_train)
    snaps_val       = list(getattr(data_cfg, "snaps_val",     snaps_train[:1]))

    use_strain      = bool(getattr(model_cfg, "use_strain",     True))
    use_invariants  = bool(getattr(model_cfg, "use_invariants", False))
    use_velocity    = bool(getattr(model_cfg, "use_velocity",   True))
    n_shell         = int(getattr(model_cfg,  "n_shell",        0))
    env_pool_mode   = str(getattr(model_cfg,  "env_pool_mode",  "mean_var"))
    hidden_dim      = int(getattr(model_cfg,  "hidden_dim",     64))
    n_layers        = int(getattr(model_cfg,  "n_layers",       3))
    cnn_ckpt_path   = getattr(model_cfg, "cnn_checkpoint", None)

    n_steps         = int(getattr(train_cfg, "n_steps",       5000))
    lr_val          = float(getattr(train_cfg, "lr",           3e-4))
    weight_decay    = float(getattr(train_cfg, "weight_decay", 1e-4))
    warmup          = int(getattr(train_cfg,   "warmup_steps",  200))
    log_every       = int(getattr(train_cfg,   "log_every",      50))
    save_every      = int(getattr(train_cfg,   "save_every",    500))
    seed            = int(getattr(train_cfg,   "seed",            0))

    sc_boost        = float(getattr(train_cfg, "loss_sc_boost",      1.0))
    density_boost   = float(getattr(train_cfg, "loss_density_boost", 0.0))
    density_gamma   = float(getattr(train_cfg, "loss_density_gamma", 0.5))

    n_train_datasets = len(sim_ids_train) * len(snaps_train)
    logger.info(f"Pre-training: {len(sim_ids_train)} sims × {len(snaps_train)} snaps "
                f"= {n_train_datasets} datasets  (ALL particles per dataset)")
    logger.info(f"Validation:   {len(sim_ids_val)} sims × {len(snaps_val)} snaps")

    # ── Neighbour indices ─────────────────────────────────────────────────────
    neighbor_idx = get_axis_neighbor_indices(n_part)
    ext_idx, ext_off, shell_sl = None, None, ()
    if n_shell > 0:
        ext_idx, ext_off, shell_sl = get_shell_neighbor_indices(n_part, n_shell)

    # ── Optional CNN prior ────────────────────────────────────────────────────
    cnn_model, cnn_params = None, None
    IS_TWO_STAGE = cnn_ckpt_path is not None
    if IS_TWO_STAGE:
        logger.info(f"Loading CNN checkpoint: {cnn_ckpt_path}")
        cnn_model, cnn_params = _load_cnn_massres_checkpoint(cnn_ckpt_path)
        n_cnn = sum(x.size for x in jax.tree_util.tree_leaves(cnn_params))
        logger.info(f"  CNN loaded  ({n_cnn:,} params)")
    else:
        logger.info("Single-stage: MLP trains on full ΔF")

    # ── Precompute all training datasets ──────────────────────────────────────
    logger.info("Precomputing training datasets …")
    train_datasets = []
    for sim_id in sim_ids_train:
        for snap_idx in snaps_train:
            logger.info(f"  sim={sim_id}  snap={snap_idx}")
            ds = build_dataset(
                data_dir, sim_id, snap_idx,
                n_part, mesh_lr, mesh_hr, box_size,
                neighbor_idx, use_strain, use_invariants, use_velocity,
                ext_idx, ext_off, env_pool_mode, shell_sl,
                cnn_model, cnn_params,
                sc_boost, density_boost, density_gamma,
            )
            df_mag_full = float(jnp.mean(jnp.sqrt(jnp.sum(ds["delta_f"]**2, axis=-1))))
            tgt_mag     = float(jnp.mean(jnp.sqrt(jnp.sum(ds["target"]**2, axis=-1))))
            logger.info(f"    a={ds['a']:.4f}  |ΔF|={df_mag_full:.4e}  "
                        f"|target|={tgt_mag:.4e}  SC={ds['sc_frac']:.2%}")
            train_datasets.append(ds)

    # ── Precompute validation datasets ────────────────────────────────────────
    logger.info("Precomputing validation datasets …")
    val_datasets = []
    for sim_id in sim_ids_val:
        for snap_idx in snaps_val:
            logger.info(f"  sim={sim_id}  snap={snap_idx}")
            ds = build_dataset(
                data_dir, sim_id, snap_idx,
                n_part, mesh_lr, mesh_hr, box_size,
                neighbor_idx, use_strain, use_invariants, use_velocity,
                ext_idx, ext_off, env_pool_mode, shell_sl,
                cnn_model, cnn_params,
                sc_boost, density_boost, density_gamma,
            )
            val_datasets.append(ds)

    if not val_datasets:
        logger.warning("No validation datasets — val metrics will be skipped")

    # ── Model ─────────────────────────────────────────────────────────────────
    lag_model = make_lagrangian_corrector(
        hidden_dim=hidden_dim, n_layers=n_layers, output_dim=3
    )
    schedule  = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=lr_val,
        warmup_steps=warmup, decay_steps=n_steps, end_value=lr_val * 0.01,
    )
    optimizer = optax.adamw(schedule, weight_decay=weight_decay)

    # Init using first dataset
    ds0 = train_datasets[0]
    rng = jax.random.PRNGKey(seed)
    params    = lag_model.init(rng, ds0["feats"][:4], ds0["vel"][:4], jnp.array(ds0["a"]))
    opt_state = optimizer.init(params)

    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    logger.info(f"MLP: {n_params:,} params  hidden={hidden_dim}×{n_layers}")
    wandb.log({"model/n_params": n_params, "model/is_two_stage": int(IS_TWO_STAGE)}, step=0)

    train_step_fn = make_train_step(lag_model, optimizer)
    rng_np = np.random.default_rng(seed)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_r  = -np.inf
    best_params = None

    for step in range(n_steps):
        # Random dataset selection — key to multi-sim generalisation
        ds = train_datasets[rng_np.integers(0, len(train_datasets))]

        params, opt_state, loss = train_step_fn(
            params, opt_state,
            ds["feats"], ds["vel"], jnp.array(ds["a"]),
            ds["target"], ds["weights"],
        )

        if step % log_every == 0 or step == n_steps - 1:
            log = {"step": step, "train/loss": float(loss)}

            # Quick train-dataset sample metrics (last used dataset)
            pred_ds = np.asarray(jax.device_get(
                jax.jit(lag_model.apply)(
                    params, ds["feats"], ds["vel"], jnp.array(ds["a"])
                )
            ))
            cnn_ds  = ds["cnn_pred"] if ds["cnn_pred"] is not None else np.zeros_like(pred_ds)
            total_ds = cnn_ds + pred_ds
            df_ds    = np.asarray(jax.device_get(ds["delta_f"]))
            r_ds = float(np.mean([pearsonr(total_ds[:, c], df_ds[:, c])[0] for c in range(3)]))
            log["train/pearson_r_last_dataset"] = r_ds

            # Validation
            val_r = float("nan")
            if val_datasets:
                vlog, val_r = validate(lag_model, params, val_datasets)
                log.update(vlog)

                if val_r > best_val_r:
                    best_val_r  = val_r
                    best_params = hk.data_structures.to_mutable_dict(params)
                    with open(out_dir / "best_params.pkl", "wb") as fh:
                        pickle.dump(best_params, fh)

            wandb.log(log, step=step)
            logger.info(
                f"step {step:5d} | loss={float(loss):.3e} | "
                f"train R(last)={r_ds:.3f} | "
                f"val R={val_r:.3f}  (best={best_val_r:.3f})"
            )

        if step % save_every == 0:
            with open(out_dir / "checkpoint.pkl", "wb") as fh:
                pickle.dump({
                    "params":    hk.data_structures.to_mutable_dict(params),
                    "model_cfg": vars(model_cfg),
                    "data_cfg":  vars(data_cfg),
                    "step":      step,
                }, fh)

    # ── Final save ────────────────────────────────────────────────────────────
    final_p = hk.data_structures.to_mutable_dict(params)
    with open(out_dir / "final_params.pkl", "wb") as fh:
        pickle.dump(final_p, fh)

    # Full checkpoint — same format as train_subregion_forceres.py so the
    # fine-tuner can load it directly
    ckpt_full = {
        "params":       final_p,
        "model_cfg":    vars(model_cfg),
        "data_cfg":     vars(data_cfg),
        "cnn_ckpt":     cnn_ckpt_path,
        "is_two_stage": IS_TWO_STAGE,
    }
    with open(out_dir / "checkpoint.pkl", "wb") as fh:
        pickle.dump(ckpt_full, fh)

    logger.info(f"Pre-training done → {out_dir}")
    logger.info(f"Best val R: {best_val_r:.4f}")
    logger.info("\nTo use as MLP prior in fine-tuning, set in finetune_forceres.yaml:")
    logger.info(f"  model:\n    mlp_checkpoint: {out_dir / 'checkpoint.pkl'}")
    wandb.finish()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Multi-simulation pre-training for force-res MLP"
    )
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train(args.config)
