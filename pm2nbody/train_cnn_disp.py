"""
train_cnn_disp.py — CNN-WST displacement field super-resolution.

Architecture
------------
Same CNN-WST as the force pipeline, but trained to minimise

    L = Σ_i w_i · ||∇ΔΦ_CNN(x_LR_i) − ΔΨ_i||²

where ΔΨ_i = Ψ_HR(q_i) − Ψ_LR_i  (HR minus LR displacement at LR Lagrangian q).

The CNN takes the LR density grid (+ PM potential if use_pm_potential=True) and
outputs a scalar potential correction ΔΦ per particle.  Its gradient w.r.t.
particle positions is the displacement correction.

CNN type choices (config model.type):
  "cnn"     — plain CNN, fast
  "cnn_wst" — CNN + Wavelet Scattering Transform (default, best accuracy)

Output checkpoint is compatible with train_lag_disp.py via cnn_checkpoint.

Usage
-----
  python train_cnn_disp.py --config configs/cnn_disp.yaml
"""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.85"

import sys
import yaml
import pickle
import logging
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import optax
import wandb

from jaxpm.kernels import fftk
from jaxpm.pm import get_delta, potential_kgrid_to_force_at_pos
from jaxpm.nn import CNN

from wst import WaveletScatteringTransform

# Re-use data loading from force pipeline
from train_lag_force import load_snapshot, compute_sample_weights
# Displacement pair from displacement pipeline
from train_lag_disp import compute_displacement_pair

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")


# ==============================================================================
# Grid features  (density + optional PM potential)
# ==============================================================================

def build_grid_data(pos_lr_mod: jnp.ndarray, mesh_lr: int, use_pm_potential: bool):
    """
    Build [mesh_lr, mesh_lr, mesh_lr, C] grid input for the CNN.

    C = 2  if use_pm_potential  (density δ  +  PM potential Φ_PM)
    C = 1  otherwise            (density δ only)
    """
    mesh_shape = (mesh_lr,) * 3
    delta      = get_delta(pos_lr_mod, mesh_shape)          # [M,M,M]

    if use_pm_potential:
        delta_k = jnp.fft.rfftn(delta)
        kvec    = fftk(mesh_shape)
        _, pm_pot = potential_kgrid_to_force_at_pos(
            delta_k, pos_lr_mod, kvec, return_potential=True
        )
        # pm_pot is already the potential grid [M, M, M] — stack directly
        return jnp.stack([pm_pot, delta], axis=-1)          # [M,M,M, 2]

    return delta[..., None]                                  # [M,M,M, 1]


# ==============================================================================
# Model factory
# ==============================================================================

def build_cnn_model(cfg):
    """
    Return a haiku-transformed model (cnn or cnn_wst).
    Output: [N, 1]  per-particle scalar potential correction ΔΦ.
    """
    model_type  = cfg.type          # "cnn" or "cnn_wst"
    input_dim   = cfg.input_dim     # must match C from build_grid_data
    wst_J       = getattr(cfg, "wst_J", 2)
    wst_L       = getattr(cfg, "wst_L", 4)

    def _model(x, positions, a, velocities=None):
        if model_type == "cnn_wst":
            wst   = WaveletScatteringTransform(J=wst_J, L=wst_L, normalize=True)
            x_in  = wst(x)
            actual_input_dim = x.shape[-1] + wst_J * wst_L
        else:
            x_in  = x
            actual_input_dim = input_dim

        cnn = CNN(
            channels_hidden_dim        = cfg.channels_hidden_dim,
            n_convolutions             = cfg.n_convolutions,
            n_fully_connected          = cfg.n_fully_connected,
            input_dim                  = actual_input_dim,
            output_dim                 = 1,
            kernel_size                = getattr(cfg, "kernel_size",        3),
            pad_periodic               = getattr(cfg, "pad_periodic",    True),
            embed_globals              = getattr(cfg, "embed_globals",   True),
            n_globals_embedding        = getattr(cfg, "n_globals_embedding", 2),
            globals_embedding_dim      = getattr(cfg, "globals_embedding_dim", 8),
            global_conditioning        = getattr(cfg, "global_conditioning", None),
            use_attention_interpolation= getattr(cfg, "use_attention_interpolation", False),
            add_particle_velocities    = getattr(cfg, "add_particle_velocities", False),
        )
        return cnn(
            x=x_in,
            positions=positions,
            global_features=jnp.atleast_1d(jnp.array(a)),
            velocities=velocities,
        )

    return hk.without_apply_rng(hk.transform(_model))


# ==============================================================================
# Loss  (gradient of CNN potential → displacement correction)
# ==============================================================================

def make_loss_fn(model, mesh_lr: int):
    """
    Returns a JIT-compiled loss function.

    Loss = weighted MSE( ∇ΔΦ_CNN(x_LR), ΔΨ_target )

    The gradient of the scalar potential output w.r.t. particle positions is the
    per-particle displacement correction — curl-free by construction, matching the
    dominant 1LPT contribution.
    """
    @jax.jit
    def loss_fn(
        params:        hk.Params,
        grid_data:     jnp.ndarray,   # [M,M,M, C]
        pos_lr_t:      jnp.ndarray,   # [N, 3]  in mesh units
        a_val:         float,
        target_disp:   jnp.ndarray,   # [N, 3]  ΔΨ target in mesh units
        weights:       jnp.ndarray,   # [N]     per-particle loss weights
    ) -> jnp.ndarray:
        # ∇ΔΦ via autodiff: compute sum(ΔΦ) and differentiate w.r.t. positions
        def phi_sum(pos):
            return jnp.sum(model.apply(params, grid_data, pos, a_val)[:, 0])

        disp_pred = jax.grad(phi_sum)(pos_lr_t)             # [N, 3]
        sq_err    = jnp.sum((disp_pred - target_disp) ** 2, axis=-1)  # [N]
        return jnp.mean(weights * sq_err)

    return loss_fn


# ==============================================================================
# Train step
# ==============================================================================

def make_train_step(model, optimizer, mesh_lr):
    loss_fn = make_loss_fn(model, mesh_lr)

    @jax.jit
    def train_step(params, opt_state, grid_data, pos_lr_t, a_val, target, weights):
        loss, grads = jax.value_and_grad(loss_fn)(
            params, grid_data, pos_lr_t, a_val, target, weights
        )
        updates, opt_state_new = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state_new, loss

    return train_step


# ==============================================================================
# Displacement metrics
# ==============================================================================

def disp_mse(pred, target):
    return float(jnp.mean((pred - target) ** 2))

def pearson_r(x, y):
    xc = x - x.mean(); yc = y - y.mean()
    return float(jnp.sum(xc*yc) / (jnp.sqrt(jnp.sum(xc**2)*jnp.sum(yc**2)) + 1e-12))

def compute_cnn_pred(model, params, grid_data, pos_lr_t, a_val):
    """Returns ∇ΔΦ_CNN at all particles [N, 3]."""
    def phi_sum(pos):
        return jnp.sum(model.apply(params, grid_data, pos, a_val)[:, 0])
    return jax.grad(phi_sum)(pos_lr_t)


def val_metrics(pred_fn, params, data_dir, sim_id, snap_ids, mesh_lr, mesh_hr,
                box_size, n_part, use_pm_potential, prefix="val/"):
    """
    pred_fn: callable(params, grid_data, pos_lr, a) → [N, 3]
             Should be a JIT-compiled lambda that closes over the model.
    """
    mses = []
    log  = {}
    for sid in snap_ids:
        pos_lr, vel_lr, pos_hr, a = load_snapshot(
            data_dir, sim_id, sid, mesh_lr, mesh_hr, box_size
        )
        pos_mod   = jnp.mod(pos_lr, mesh_lr)
        gd        = build_grid_data(pos_mod, mesh_lr, use_pm_potential)
        _, _, dps = compute_displacement_pair(pos_lr, pos_hr, n_part, mesh_lr, mesh_hr)

        pred  = pred_fn(params, gd, pos_lr, a)
        mse   = disp_mse(pred, dps)
        r_x   = pearson_r(pred[:, 0], dps[:, 0])
        r_y   = pearson_r(pred[:, 1], dps[:, 1])
        r_z   = pearson_r(pred[:, 2], dps[:, 2])
        mses.append(mse)
        log[f"{prefix}snap{sid}/disp_mse"]       = mse
        log[f"{prefix}snap{sid}/pearson_r_mean"] = (r_x + r_y + r_z) / 3.0
        del pos_lr, vel_lr, pos_hr, gd, dps, pred

    log[f"{prefix}disp_mse_mean"] = float(np.mean(mses))
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

    # ── WandB ─────────────────────────────────────────────────────────────────
    wandb.init(
        project = wb_cfg.get("project", "pm2nbody_cnn_disp"),
        name    = getattr(exp_cfg, "name", None),
        tags    = wb_cfg.get("tags", []),
        config  = cfg,
    )
    out_dir = Path(getattr(exp_cfg, "output_dir", "runs/cnn_disp")) / wandb.run.name
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, out_dir / "config.yaml")

    # ── Config ────────────────────────────────────────────────────────────────
    mesh_lr    = int(data_cfg.mesh_lr)
    mesh_hr    = int(data_cfg.mesh_hr)
    box_size   = float(data_cfg.box_size)
    n_part     = int(data_cfg.n_particles)
    data_dir   = Path(data_cfg.data_dir)
    snap_train = int(data_cfg.snap_train)
    sim_train  = int(getattr(data_cfg, "sim_id_train", 0))
    sim_val    = int(getattr(data_cfg, "sim_id_val",   1))
    snaps_val  = list(getattr(data_cfg, "snaps_val", [snap_train]))

    use_pm_potential = bool(getattr(model_cfg, "use_pm_potential", True))
    n_steps     = int(getattr(train_cfg, "n_steps",       2000))
    lr_val      = float(getattr(train_cfg, "lr",           1e-4))
    wd          = float(getattr(train_cfg, "weight_decay", 1e-4))
    warmup      = int(getattr(train_cfg, "warmup_steps",    100))
    log_every   = int(getattr(train_cfg, "log_every",        50))
    save_every  = int(getattr(train_cfg, "save_every",       500))
    seed        = int(getattr(train_cfg, "seed",               0))

    loss_sc_boost      = float(getattr(train_cfg, "loss_sc_boost",      1.0))
    loss_density_boost = float(getattr(train_cfg, "loss_density_boost", 0.0))
    loss_density_gamma = float(getattr(train_cfg, "loss_density_gamma", 0.5))

    logger.info(f"CNN type: {model_cfg.type}  |  use_pm_potential: {use_pm_potential}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_cnn_model(model_cfg)

    # ── Training snapshot ─────────────────────────────────────────────────────
    logger.info(f"Loading training snapshot  sim={sim_train}  snap={snap_train}")
    pos_lr_t, vel_lr_t, pos_hr_t, a_train = load_snapshot(
        data_dir, sim_train, snap_train, mesh_lr, mesh_hr, box_size
    )
    if pos_hr_t is None:
        raise RuntimeError("HR positions required — check data_dir and mesh_hr in config.")
    logger.info(f"  a={a_train:.4f}  N_lr={pos_lr_t.shape[0]:,}  N_hr={pos_hr_t.shape[0]:,}")

    # Grid data (build once; positions don't change between steps for snapshot training)
    pos_mod   = jnp.mod(pos_lr_t, mesh_lr)
    grid_data = jax.jit(build_grid_data, static_argnums=(1, 2))(
        pos_mod, mesh_lr, use_pm_potential
    )
    logger.info(f"  grid_data shape: {grid_data.shape}")

    # Displacement target
    logger.info("Computing displacement pair …")
    psi_lr_tr, psi_hr_tr, delta_psi_tr = compute_displacement_pair(
        pos_lr_t, pos_hr_t, n_part, mesh_lr, mesh_hr
    )
    disp_mag = float(jnp.mean(jnp.sqrt(jnp.sum(delta_psi_tr ** 2, axis=-1))))
    logger.info(f"  |ΔΨ| mean = {disp_mag:.4e}")

    # Per-particle weights
    if loss_sc_boost > 1.0 or loss_density_boost > 0.0:
        from jaxpm.lagrangian import (
            get_axis_neighbor_indices, compute_deformation_tensor
        )
        nbrs = get_axis_neighbor_indices(n_part)
        neg_idx = nbrs[:, [1, 3, 5]]
        _, det_D_np = jax.jit(
            lambda p: compute_deformation_tensor(p, neg_idx, mesh_lr)
        )(pos_lr_t)
        det_D_np = np.asarray(jax.device_get(det_D_np))
        weights_np = compute_sample_weights(
            pos_lr_t, det_D_np, mesh_lr,
            sc_boost=loss_sc_boost,
            density_boost=loss_density_boost,
            density_gamma=loss_density_gamma,
        )
        logger.info(
            f"  Sample weights: min={weights_np.min():.3f}  max={weights_np.max():.3f}"
        )
    else:
        weights_np = np.ones(pos_lr_t.shape[0], dtype=np.float32)

    weights_t = jnp.array(weights_np)

    # ── Init model ────────────────────────────────────────────────────────────
    rng    = jax.random.PRNGKey(seed)
    params = model.init(rng, grid_data, pos_lr_t, a_train)
    n_p    = sum(x.size for x in jax.tree_util.tree_leaves(params))
    logger.info(f"Parameters: {n_p:,}")

    wandb.log({
        "model/n_params":       n_p,
        "data/disp_mag_target": disp_mag,
        "data/grid_C":          int(grid_data.shape[-1]),
    }, step=0)

    # ── Optimiser ─────────────────────────────────────────────────────────────
    lr_sched  = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=lr_val,
        warmup_steps=warmup, decay_steps=n_steps, end_value=lr_val * 0.01,
    )
    optimizer  = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(lr_sched, weight_decay=wd),
    )
    opt_state  = optimizer.init(params)
    train_step = make_train_step(model, optimizer, mesh_lr)

    # JIT-compiled helpers that close over `model` (Haiku object — not a JAX array,
    # so it must be captured in the closure, never passed as a traced argument)
    _pred_jit = jax.jit(
        lambda p, gd, pos, a: compute_cnn_pred(model, p, gd, pos, a)
    )

    # ── Loop ──────────────────────────────────────────────────────────────────
    best_val  = float("inf")
    best_prms = None

    for step in range(1, n_steps + 1):
        params, opt_state, loss = train_step(
            params, opt_state,
            grid_data, pos_lr_t, a_train,
            delta_psi_tr, weights_t,
        )

        if step % log_every == 0 or step == 1:
            pred_tr  = _pred_jit(params, grid_data, pos_lr_t, a_train)
            tr_mse   = disp_mse(pred_tr, delta_psi_tr)
            r_x      = pearson_r(pred_tr[:, 0], delta_psi_tr[:, 0])
            r_y      = pearson_r(pred_tr[:, 1], delta_psi_tr[:, 1])
            r_z      = pearson_r(pred_tr[:, 2], delta_psi_tr[:, 2])

            log_dict = {
                "train/loss":           float(loss),
                "train/disp_mse":       tr_mse,
                "train/pearson_r_mean": (r_x + r_y + r_z) / 3.0,
                "train/lr":             float(lr_sched(step)),
            }
            del pred_tr

            vlog, vmse = val_metrics(
                _pred_jit, params, data_dir, sim_val, snaps_val,
                mesh_lr, mesh_hr, box_size, n_part, use_pm_potential,
            )
            log_dict.update(vlog)
            wandb.log(log_dict, step=step)

            logger.info(
                f"step {step:5d}  loss={float(loss):.4e}"
                f"  R̄={log_dict['train/pearson_r_mean']:.3f}"
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

    # ── Save ──────────────────────────────────────────────────────────────────
    if best_prms is not None:
        with open(out_dir / "best_params.pkl", "wb") as fh:
            pickle.dump(best_prms, fh)

    with open(out_dir / "final_params.pkl", "wb") as fh:
        pickle.dump(jax.device_get(hk.data_structures.to_mutable_dict(params)), fh)

    # Save full checkpoint (model def + params) compatible with eval_utils
    ckpt_full = {
        "params":    jax.device_get(hk.data_structures.to_mutable_dict(params)),
        "model_cfg": {k: getattr(model_cfg, k) for k in vars(model_cfg)
                      if not k.startswith("_")},
        "data_cfg":  {k: getattr(data_cfg,  k) for k in vars(data_cfg)
                      if not k.startswith("_")},
    }
    with open(out_dir / "checkpoint.pkl", "wb") as fh:
        pickle.dump(ckpt_full, fh)

    wandb.log({"val/best_disp_mse": best_val}, step=n_steps)
    logger.info(f"Done. best_val_mse={best_val:.4e}  output → {out_dir}")
    logger.info(f"\nTo use as CNN prior in MLP training, add to lag_disp_mlp.yaml:")
    logger.info(f"  model:\n    cnn_checkpoint: {out_dir / 'checkpoint.pkl'}")
    wandb.finish()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CNN-WST displacement super-resolution")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train(args.config)
