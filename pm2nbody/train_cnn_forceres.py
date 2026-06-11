"""
train_cnn_forceres.py — CNN-WST force-resolution correction (stage 1).

Architecture
------------
Identical CNN-WST to train_cnn_massres.py.  The difference is only in the
target: here we correct FORCE RESOLUTION (same particles, coarse vs fine PM
mesh) instead of mass resolution (LR vs HR particles).

    Target: ΔF = F_fine(pos, mesh_hr) − F_coarse(pos, mesh_lr)

The CNN is trained on the full simulation (all particles).  Its checkpoint
can then be used by train_subregion_forceres.py as a stage-1 prior:

    model:
      cnn_checkpoint: runs/cnn_forceres/<run>/checkpoint.pkl

Pipeline
--------
  Stage 1 (this script):  train_cnn_forceres.py   --config configs/cnn_forceres.yaml
  Stage 2 (MLP residual): train_subregion_forceres.py  --config configs/subregion_forceres.yaml
    set model.cnn_checkpoint in subregion_forceres.yaml to activate residual mode.

Unit convention
---------------
  - Data loaded in n_part units (pos ∈ [0, n_part)) via load_single_snapshot.
  - compute_force_pair expects mesh_lr units → convert: pos_lr = pos * mesh_lr / n_part
  - CNN grid: build_grid_data(pos % mesh_lr, mesh_lr, ...) — positions in mesh_lr units.

Usage
-----
  python train_cnn_forceres.py --config configs/cnn_forceres.yaml
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

# ── Reuse CNN infrastructure from train_cnn_massres ───────────────────────────
from train_cnn_massres import (
    build_cnn_model,         # model factory
    build_grid_data,         # [M,M,M,C] grid builder (δ + optional Φ_PM)
    compute_cnn_pred,        # ∇ΔΦ_CNN  [N, 3]
    make_train_step,         # JIT train step
    force_mse,
    pearson_r,
)

# ── Force-resolution pair (same particles, two mesh sizes) ────────────────────
from train_lag_force import compute_force_pair, compute_sample_weights

# ── Single-simulation data loader (same as train_subregion_forceres.py) ───────
from train_subregion_forceres import load_single_snapshot

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")


# ==============================================================================
# Validation helper
# ==============================================================================

def val_metrics_forceres(
    pred_jit, params, data_dir, sim_id, snap_ids,
    n_part, mesh_lr, mesh_hr, box_size, use_pm_potential,
    prefix="val/",
):
    """
    Evaluate CNN on a set of snapshots from a single simulation.

    Parameters
    ----------
    pred_jit : callable(params, grid_data, pos_lr, a) → [N, 3] force correction
    """
    scale_to_lr = float(mesh_lr) / float(n_part)
    mses  = []
    log   = {}
    for sid in snap_ids:
        pos, vel, a = load_single_snapshot(
            Path(data_dir), sim_id, sid, n_part, box_size
        )
        pos_lr  = pos * scale_to_lr          # n_part units → mesh_lr units
        pos_mod = jnp.mod(pos_lr, mesh_lr)
        gd      = jax.jit(build_grid_data, static_argnums=(1, 2))(
            pos_mod, mesh_lr, use_pm_potential
        )

        _fp = jax.jit(partial(compute_force_pair, mesh_lr=mesh_lr, mesh_hr=mesh_hr))
        _, _, df = _fp(pos_lr)

        pred = pred_jit(params, gd, pos_lr, a)
        mse  = force_mse(pred, df)
        rx   = pearson_r(pred[:, 0], df[:, 0])
        ry   = pearson_r(pred[:, 1], df[:, 1])
        rz   = pearson_r(pred[:, 2], df[:, 2])
        mses.append(mse)
        log[f"{prefix}snap{sid}/force_mse"]      = mse
        log[f"{prefix}snap{sid}/pearson_r_mean"] = (rx + ry + rz) / 3.0
        del pos, vel, gd, df, pred

    log[f"{prefix}force_mse_mean"] = float(np.mean(mses))
    return log, float(np.mean(mses))


# ==============================================================================
# Main
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
        project = wb_cfg.get("project", "pm2nbody_cnn_forceres"),
        name    = getattr(exp_cfg, "name", None),
        tags    = wb_cfg.get("tags", []),
        config  = cfg,
    )
    out_dir = Path(getattr(exp_cfg, "output_dir", "runs/cnn_forceres")) / wandb.run.name
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, out_dir / "config.yaml")

    # ── Config ────────────────────────────────────────────────────────────────
    n_part   = int(data_cfg.n_part)
    mesh_lr  = int(data_cfg.mesh_lr)
    mesh_hr  = int(data_cfg.mesh_hr)
    box_size = float(data_cfg.box_size)
    data_dir = Path(data_cfg.data_dir)

    sim_train  = int(getattr(data_cfg, "sim_id_train", 0))
    sim_val    = int(getattr(data_cfg, "sim_id_val",   1))
    # Accept either snap_train (int, backward compat) or snaps_train (list)
    _s = getattr(data_cfg, "snaps_train", None) or getattr(data_cfg, "snap_train", 5)
    snaps_train = [int(_s)] if isinstance(_s, (int, float)) else [int(x) for x in _s]
    snaps_val   = list(getattr(data_cfg, "snaps_val", snaps_train[:1]))

    use_pm_potential = bool(getattr(model_cfg, "use_pm_potential", True))
    n_steps      = int(getattr(train_cfg, "n_steps",       2000))
    lr_val       = float(getattr(train_cfg, "lr",           1e-4))
    wd           = float(getattr(train_cfg, "weight_decay", 1e-4))
    warmup       = int(getattr(train_cfg, "warmup_steps",    100))
    log_every    = int(getattr(train_cfg, "log_every",        50))
    save_every   = int(getattr(train_cfg, "save_every",       500))
    seed         = int(getattr(train_cfg, "seed",               0))

    loss_sc_boost  = float(getattr(train_cfg, "loss_sc_boost",      1.0))
    loss_dens_boost= float(getattr(train_cfg, "loss_density_boost", 0.0))
    loss_dens_gamma= float(getattr(train_cfg, "loss_density_gamma", 0.5))

    logger.info(f"Force-res CNN  n_part={n_part}  mesh_lr={mesh_lr}  mesh_hr={mesh_hr}")
    logger.info(f"CNN type: {model_cfg.type}  use_pm_potential: {use_pm_potential}")

    # ── Unit conversion: n_part → mesh_lr ────────────────────────────────────
    # compute_force_pair and build_grid_data expect positions in mesh_lr units.
    scale_to_lr = float(mesh_lr) / float(n_part)
    logger.info(f"Position scale: n_part={n_part} → mesh_lr={mesh_lr}  factor={scale_to_lr:.4f}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_cnn_model(model_cfg)

    # ── Precompute all training snapshots ─────────────────────────────────────
    # Supports snaps_train as a list for temporal generalization.
    # Each dataset stores (grid_data, pos_lr, delta_f, weights, a).
    _fp_jit = jax.jit(partial(compute_force_pair, mesh_lr=mesh_lr, mesh_hr=mesh_hr))
    _gd_jit = jax.jit(build_grid_data, static_argnums=(1, 2))

    logger.info(f"Loading {len(snaps_train)} training snapshot(s) from sim={sim_train} …")
    train_datasets = []
    for snap_idx in snaps_train:
        logger.info(f"  snap={snap_idx}")
        pos, vel, a_snap = load_single_snapshot(
            data_dir, sim_train, snap_idx, n_part, box_size
        )
        pos_lr  = pos * scale_to_lr
        pos_mod = jnp.mod(pos_lr, mesh_lr)
        gd      = _gd_jit(pos_mod, mesh_lr, use_pm_potential)
        _, f_fine, delta_f = _fp_jit(pos_lr)
        w = compute_sample_weights(
            pos_lr, np.ones(pos.shape[0], dtype=np.float32),
            mesh_lr,
            sc_boost=loss_sc_boost,
            density_boost=loss_dens_boost,
            density_gamma=loss_dens_gamma,
        )
        df_mag = float(jnp.mean(jnp.sqrt(jnp.sum(delta_f**2, axis=-1))))
        f_mag  = float(jnp.mean(jnp.sqrt(jnp.sum(f_fine**2,  axis=-1))))
        logger.info(f"    a={a_snap:.3f}  |ΔF|/|F|={df_mag/f_mag:.3f}"
                    f"  grid={gd.shape}")
        train_datasets.append(dict(
            grid_data=gd, pos_lr=pos_lr, delta_f=delta_f,
            weights=jnp.array(w), a=a_snap, snap_idx=snap_idx,
        ))
        del pos, vel, f_fine

    rng_np = np.random.default_rng(seed)

    # ── Init model with first dataset ─────────────────────────────────────────
    d0     = train_datasets[0]
    rng    = jax.random.PRNGKey(seed)
    params = model.init(rng, d0["grid_data"], d0["pos_lr"], d0["a"])
    n_p    = sum(x.size for x in jax.tree_util.tree_leaves(params))
    logger.info(f"Parameters: {n_p:,}")

    wandb.log({"model/n_params": n_p, "data/n_train_snaps": len(snaps_train)}, step=0)

    # ── Optimiser ─────────────────────────────────────────────────────────────
    lr_sched  = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=lr_val,
        warmup_steps=warmup, decay_steps=n_steps, end_value=lr_val * 0.01,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(lr_sched, weight_decay=wd),
    )
    opt_state  = optimizer.init(params)
    train_step = make_train_step(model, optimizer, mesh_lr)

    _pred_jit = jax.jit(
        lambda p, gd, pos, a: compute_cnn_pred(model, p, gd, pos, a)
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val  = float("inf")
    best_prms = None

    for step in range(1, n_steps + 1):
        # Pick random training snapshot each step (or cycle if only one)
        ds = train_datasets[rng_np.integers(0, len(train_datasets))]

        params, opt_state, loss = train_step(
            params, opt_state,
            ds["grid_data"], ds["pos_lr"], ds["a"],
            ds["delta_f"], ds["weights"],
        )

        if step % log_every == 0 or step == 1:
            # Evaluate on ALL training snapshots, report mean
            tr_mses, tr_rs = [], []
            for ds_eval in train_datasets:
                pred_tr = _pred_jit(params, ds_eval["grid_data"],
                                    ds_eval["pos_lr"], ds_eval["a"])
                tr_mses.append(force_mse(pred_tr, ds_eval["delta_f"]))
                rx = pearson_r(pred_tr[:, 0], ds_eval["delta_f"][:, 0])
                ry = pearson_r(pred_tr[:, 1], ds_eval["delta_f"][:, 1])
                rz = pearson_r(pred_tr[:, 2], ds_eval["delta_f"][:, 2])
                tr_rs.append((rx + ry + rz) / 3.0)
                del pred_tr
            r_mean = float(np.mean(tr_rs))

            log_dict = {
                "train/loss":           float(loss),
                "train/force_mse":      float(np.mean(tr_mses)),
                "train/pearson_r_mean": r_mean,
                "train/lr":             float(lr_sched(step)),
            }

            vlog, vmse = val_metrics_forceres(
                _pred_jit, params, data_dir, sim_val, snaps_val,
                n_part, mesh_lr, mesh_hr, box_size, use_pm_potential,
            )
            log_dict.update(vlog)
            wandb.log(log_dict, step=step)

            logger.info(
                f"step {step:5d}  loss={float(loss):.4e}"
                f"  R̄={r_mean:.4f}"
                f"  val_mse={vmse:.4e}"
            )

            if vmse < best_val:
                best_val  = vmse
                best_prms = jax.device_get(hk.data_structures.to_mutable_dict(params))

        if step % save_every == 0:
            ckpt = out_dir / f"params_step{step:05d}.pkl"
            with open(ckpt, "wb") as fh:
                pickle.dump(jax.device_get(hk.data_structures.to_mutable_dict(params)), fh)
            logger.info(f"  checkpoint → {ckpt}")

    # ── Save — checkpoint format compatible with train_subregion_forceres.py ──
    if best_prms is not None:
        with open(out_dir / "best_params.pkl", "wb") as fh:
            pickle.dump(best_prms, fh)

    final_params = jax.device_get(hk.data_structures.to_mutable_dict(params))
    with open(out_dir / "final_params.pkl", "wb") as fh:
        pickle.dump(final_params, fh)

    # Full checkpoint — same format as train_cnn_massres.py so _load_cnn_massres_checkpoint works
    ckpt_full = {
        "params":    final_params,
        "model_cfg": {k: getattr(model_cfg, k) for k in vars(model_cfg)
                      if not k.startswith("_")},
        "data_cfg":  {k: getattr(data_cfg,  k) for k in vars(data_cfg)
                      if not k.startswith("_")},
    }
    with open(out_dir / "checkpoint.pkl", "wb") as fh:
        pickle.dump(ckpt_full, fh)

    wandb.log({"val/best_force_mse": best_val}, step=n_steps)
    logger.info(f"Done. best_val_mse={best_val:.4e}  output → {out_dir}")
    logger.info("\nTo use as CNN prior in subregion training, add to subregion_forceres.yaml:")
    logger.info(f"  model:\n    cnn_checkpoint: {out_dir / 'checkpoint.pkl'}")
    wandb.finish()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="CNN-WST force-resolution correction stage 1"
    )
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train(args.config)
