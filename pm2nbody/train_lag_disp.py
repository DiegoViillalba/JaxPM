"""
train_lag_disp.py — Single-snapshot Lagrangian displacement super-resolution.

Experiment design
-----------------
Given a LR simulation snapshot at scale factor a, learn a per-particle
displacement correction ΔΨ that maps LR particle positions to their
HR counterparts:

  Standalone:   target = Ψ_HR(q_i) − Ψ_LR_i
  Residual:     target = Ψ_HR(q_i) − Ψ_LR_i − ΔΨ_CNN

where:
  Ψ_LR_i     = x_LR_i − q_i          LR displacement from Lagrangian origin
  Ψ_HR(q_i)  = HR displacement field sampled at LR Lagrangian position q_i
              = x_HR_j − q_i          (j = HR particle at same Lagrangian index)
  ΔΨ_CNN     = +∇ΔΦ_CNN(x_LR)       CNN-WST scalar potential → gradient

After applying the correction:
  x_corrected_i = x_LR_i + ΔΨ_pred_i ≈ x_HR(q_i)

Physical motivation
-------------------
The LR PM simulation integrates gravity on a coarse mesh, missing small-scale
force fluctuations.  The resulting displacement field is a smoothed version of
the HR displacement field — particle positions are biased toward the large-scale
flow and miss halo-scale displacements.

The deformation tensor D encodes the LOCAL LAGRANGIAN ENVIRONMENT that
determines how much a particle deviates from the mean-field flow:
  E = D − I,  det(D) < 0  ↔  multi-streaming / shell-crossing

Lagrangian features are constant through the simulation (indexed by q, not x)
and capture information invisible in the Eulerian density field.

Key assumptions
---------------
- HR particles are stored in Lagrangian C-order (particle at Lagrangian index
  (ix, iy, iz) is at array index ix * N_hr^2 + iy * N_hr + iz).
  This is standard for PM/N-body codes with regular ICs.
- n_particles_lr = mesh_lr and n_particles_hr = mesh_hr (standard config).
- pos_lr and pos_hr from load_snapshot are both in mesh_lr units.

Usage
-----
  python train_lag_disp.py --config configs/lag_disp_mlp.yaml
  python train_lag_disp.py --config configs/lag_disp_cnn_wst.yaml
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
from jaxpm.lagrangian import (
    get_axis_neighbor_indices,
    get_shell_neighbor_indices,
    compute_deformation_features,
    compute_deformation_tensor,
    make_lagrangian_corrector,
)

# Re-use data loading and generic helpers from the force pipeline
from train_lag_force import (
    load_snapshot,
    snapshot_features,
    make_train_step,
    compute_sample_weights,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")


# ==============================================================================
# 1. Lagrangian positions
# ==============================================================================

def get_lagrangian_positions(n_part: int, mesh_lr: int) -> jnp.ndarray:
    """
    Regular Lagrangian grid: q_i = (ix, iy, iz) * (mesh_lr / n_part)  in mesh_lr units.

    For n_part = mesh_lr = 128: q_i = (ix, iy, iz) ∈ {0, 1, …, 127}³.
    Particles are assumed to be stored in C-order (ix outermost, iz innermost).

    Returns
    -------
    q : [n_part³, 3]  float32, in mesh_lr units
    """
    spacing = float(mesh_lr) / float(n_part)
    coords  = jnp.arange(n_part, dtype=jnp.float32) * spacing
    ix, iy, iz = jnp.meshgrid(coords, coords, coords, indexing="ij")
    return jnp.stack([ix.ravel(), iy.ravel(), iz.ravel()], axis=-1)   # [N, 3]


# ==============================================================================
# 2. Displacement pair computation
# ==============================================================================

def compute_displacement_pair(
    pos_lr: jnp.ndarray,
    pos_hr: jnp.ndarray,
    n_part: int,
    mesh_lr: int,
    mesh_hr: int,
) -> tuple:
    """
    Compute (Ψ_LR, Ψ_HR_at_q, ΔΨ) for one snapshot, all in mesh_lr units.

    Convention
    ----------
    pos_lr and pos_hr come from load_snapshot() → both in mesh_lr units.
    HR particles are assumed in Lagrangian C-order; LR particle i maps to
    HR particle (stride*ix, stride*iy, stride*iz) where stride = mesh_hr // n_part.

    Parameters
    ----------
    pos_lr  : [N_lr, 3]   LR positions in mesh_lr units
    pos_hr  : [N_hr, 3]   HR positions in mesh_lr units (from load_snapshot)
    n_part  : int         LR particles per dimension  (= mesh_lr typically)
    mesh_lr : int
    mesh_hr : int

    Returns
    -------
    psi_lr      : [N_lr, 3]   LR displacements  Ψ_LR = x_LR − q_LR
    psi_hr_at_q : [N_lr, 3]   HR displacements sampled at LR Lagrangian positions
    delta_psi   : [N_lr, 3]   = psi_hr_at_q − psi_lr  (correction target)
    """
    n_part_hr = mesh_hr          # HR particles per dimension (standard: n_part_hr = mesh_hr)
    stride    = mesh_hr // n_part

    # Lagrangian positions in mesh_lr units
    q_lr   = get_lagrangian_positions(n_part,    mesh_lr)   # [N_lr, 3]
    q_hr   = get_lagrangian_positions(n_part_hr, mesh_lr)   # [N_hr, 3]

    # Displacements in mesh_lr units
    psi_lr = pos_lr - q_lr                                  # [N_lr, 3]
    psi_hr = pos_hr - q_hr                                  # [N_hr, 3]

    # Sample HR displacement at LR Lagrangian positions via stride subsampling.
    # This is exact (no interpolation error) because LR Lagrangian positions are
    # a regular sub-grid of the HR Lagrangian grid.
    psi_hr_grid = psi_hr.reshape(n_part_hr, n_part_hr, n_part_hr, 3)
    psi_hr_at_q = psi_hr_grid[::stride, ::stride, ::stride, :]   # [n_part, n_part, n_part, 3]
    psi_hr_at_q = psi_hr_at_q.reshape(-1, 3)                     # [N_lr, 3]

    delta_psi = psi_hr_at_q - psi_lr                             # [N_lr, 3]
    return psi_lr, psi_hr_at_q, delta_psi


# ==============================================================================
# 3. CNN displacement correction
# ==============================================================================

def compute_cnn_disp_correction(
    cnn_model,
    cnn_params,
    pos_lr_t:  jnp.ndarray,
    vel_lr_t:  jnp.ndarray,
    a:         float,
    mesh_lr:   int,
) -> jnp.ndarray:
    """
    CNN-WST displacement correction  ΔΨ_CNN = +∇ΔΦ_CNN(x_LR).

    Architecture is identical to the force-correction CNN: the model outputs
    a scalar potential correction ΔΦ per particle, and we take its gradient
    w.r.t. particle positions.  For displacement the gradient captures the
    curl-free (irrotational) part of ΔΨ — physically the dominant contribution
    in perturbation theory (1LPT: Ψ = −∇φ).

    Sign convention: same as pm.py's get_patched_transformer_force.
    +∇ΔΦ_CNN → additive correction to LR positions.

    Returns [N, 3] in mesh_lr units.
    """
    pos_lr_mod = jnp.mod(pos_lr_t, mesh_lr)
    delta      = get_delta(pos_lr_mod, (mesh_lr,) * 3)
    delta_lr_k = jnp.fft.rfftn(delta)
    kvec_lr    = fftk((mesh_lr,) * 3)

    _, pm_pot  = potential_kgrid_to_force_at_pos(
        delta_lr_k, pos_lr_mod, kvec_lr, return_potential=True
    )
    grid_data = jnp.stack([pm_pot, delta], axis=-1)
    vel_sg    = jax.lax.stop_gradient(vel_lr_t)

    def phi_sum(pos):
        return jnp.sum(cnn_model.apply(cnn_params, grid_data, pos, a, vel_sg)[:, 0])

    return jax.grad(phi_sum)(pos_lr_t)   # [N, 3]


# ==============================================================================
# 4. Displacement metrics
# ==============================================================================

def pearson_r(x: jnp.ndarray, y: jnp.ndarray) -> float:
    x_c = x - x.mean();  y_c = y - y.mean()
    return float(jnp.sum(x_c * y_c) /
                 (jnp.sqrt(jnp.sum(x_c ** 2) * jnp.sum(y_c ** 2)) + 1e-12))


def compute_disp_metrics(
    pred:     jnp.ndarray,
    target:   jnp.ndarray,
    det_D:    np.ndarray,
    prefix:   str = "",
) -> dict:
    """
    Displacement regression metrics.

    Reported per component (x/y/z), aggregate MSE, fractional MSE,
    Pearson R, and shell-crossing fraction from the Jacobian determinant.
    """
    err       = pred - target
    target_sq = jnp.sum(target ** 2, axis=-1)

    mse_full = float(jnp.mean(err ** 2))
    mse_x    = float(jnp.mean(err[:, 0] ** 2))
    mse_y    = float(jnp.mean(err[:, 1] ** 2))
    mse_z    = float(jnp.mean(err[:, 2] ** 2))
    frac_mse = float(jnp.mean(jnp.sum(err ** 2, axis=-1) / (target_sq + 1e-12)))
    r_x      = pearson_r(pred[:, 0], target[:, 0])
    r_y      = pearson_r(pred[:, 1], target[:, 1])
    r_z      = pearson_r(pred[:, 2], target[:, 2])
    sc_frac  = float(np.mean(det_D < 0.0))
    psi_mag  = float(jnp.mean(jnp.sqrt(target_sq)))

    p = prefix
    return {
        f"{p}disp_mse":        mse_full,
        f"{p}disp_mse_x":      mse_x,
        f"{p}disp_mse_y":      mse_y,
        f"{p}disp_mse_z":      mse_z,
        f"{p}frac_mse":        frac_mse,
        f"{p}pearson_r_x":     r_x,
        f"{p}pearson_r_y":     r_y,
        f"{p}pearson_r_z":     r_z,
        f"{p}pearson_r_mean":  (r_x + r_y + r_z) / 3.0,
        f"{p}target_disp_mag": psi_mag,
        f"{p}shell_cross_frac": sc_frac,
    }


# ==============================================================================
# 5. Main training loop
# ==============================================================================

def train(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    exp_cfg   = SimpleNamespace(**cfg.get("experiment",  {}))
    data_cfg  = SimpleNamespace(**cfg["data"])
    model_cfg = SimpleNamespace(**cfg["model"])
    train_cfg = SimpleNamespace(**cfg["training"])
    wb_cfg    = cfg.get("wandb", {})

    # ── WandB ─────────────────────────────────────────────────────────────────
    wandb.init(
        project = wb_cfg.get("project", "pm2nbody_lag_disp"),
        name    = getattr(exp_cfg, "name", None),
        tags    = wb_cfg.get("tags", []),
        config  = cfg,
    )

    out_dir = Path(getattr(exp_cfg, "output_dir", "runs/lag_disp")) / wandb.run.name
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, out_dir / "config.yaml")

    # ── Config extraction ─────────────────────────────────────────────────────
    mesh_lr        = int(data_cfg.mesh_lr)
    mesh_hr        = int(data_cfg.mesh_hr)
    box_size       = float(data_cfg.box_size)
    n_part         = int(data_cfg.n_particles)
    data_dir       = Path(data_cfg.data_dir)
    snap_train     = int(data_cfg.snap_train)
    sim_train      = int(getattr(data_cfg, "sim_id_train", 0))
    sim_val        = int(getattr(data_cfg, "sim_id_val",   1))
    snaps_val      = list(getattr(data_cfg, "snaps_val",   [snap_train]))

    use_strain     = bool(getattr(model_cfg, "use_strain",     True))
    use_invariants = bool(getattr(model_cfg, "use_invariants", False))
    use_velocity   = bool(getattr(model_cfg, "use_velocity",   True))
    n_shell        = int(getattr(model_cfg, "n_shell",        0))
    env_pool_mode  = str(getattr(model_cfg, "env_pool_mode",  "mean_var"))
    hidden_dim     = int(getattr(model_cfg, "hidden_dim", 64))
    n_layers       = int(getattr(model_cfg, "n_layers",   3))

    n_steps      = int(getattr(train_cfg, "n_steps",       500))
    lr_val       = float(getattr(train_cfg, "lr",           3e-4))
    weight_decay = float(getattr(train_cfg, "weight_decay", 1e-4))
    warmup       = int(getattr(train_cfg, "warmup_steps",    50))
    log_every    = int(getattr(train_cfg, "log_every",       25))
    save_every   = int(getattr(train_cfg, "save_every",     100))
    seed         = int(getattr(train_cfg, "seed",             0))

    loss_sc_boost      = float(getattr(train_cfg, "loss_sc_boost",      1.0))
    loss_density_boost = float(getattr(train_cfg, "loss_density_boost", 0.0))
    loss_density_gamma = float(getattr(train_cfg, "loss_density_gamma", 0.5))
    use_weighted_loss  = (loss_sc_boost > 1.0) or (loss_density_boost > 0.0)

    logger.info(
        f"Loss weighting: sc_boost={loss_sc_boost}  "
        f"density_boost={loss_density_boost}  density_gamma={loss_density_gamma}  "
        f"({'active' if use_weighted_loss else 'uniform'})"
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    # output_dim = 3: predict ΔΨ directly as a 3-vector (not potential mode)
    lag_model = make_lagrangian_corrector(
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        output_dim=3,
    )
    logger.info(f"Model: hidden_dim={hidden_dim}  n_layers={n_layers}  output_dim=3 (ΔΨ)")

    # ── Optional CNN-WST residual mode ────────────────────────────────────────
    cnn_model, cnn_params = None, None
    cnn_ckpt = getattr(model_cfg, "cnn_checkpoint", None)
    if cnn_ckpt:
        logger.info(f"Loading CNN-WST checkpoint: {cnn_ckpt}")
        try:
            import eval_utils as eu
            ri = eu.load_run(Path(cnn_ckpt))
            cnn_model, cnn_params = ri["model"], ri["params"]
            logger.info("  CNN loaded successfully.")
        except Exception as e:
            logger.error(f"Could not load CNN checkpoint: {e}")

    if cnn_model is not None:
        _apply_cnn = jax.jit(
            lambda pos, vel, a: compute_cnn_disp_correction(
                cnn_model, cnn_params, pos, vel, a, mesh_lr
            )
        )
    else:
        _apply_cnn = None

    # ── Displacement pair helpers ─────────────────────────────────────────────
    _disp_pair = jax.jit(
        lambda pos_lr, pos_hr: compute_displacement_pair(
            pos_lr, pos_hr, n_part, mesh_lr, mesh_hr
        )
    )

    # ── Lagrangian structure ──────────────────────────────────────────────────
    logger.info(f"Precomputing Lagrangian neighbours  {n_part}³ = {n_part**3:,} particles")
    neighbor_idx = get_axis_neighbor_indices(n_part)

    ext_neighbor_idx, ext_offsets, shell_slices = None, None, ()
    if n_shell > 0:
        K = (2 * n_shell + 1) ** 3 - 1
        logger.info(f"Extended neighbourhood: n_shell={n_shell}  K={K}  pool='{env_pool_mode}'")
        ext_neighbor_idx, ext_offsets, shell_slices = get_shell_neighbor_indices(n_part, n_shell)

    # ── Training snapshot ─────────────────────────────────────────────────────
    logger.info(f"Loading training snapshot  sim={sim_train}  snap={snap_train}")
    pos_lr_t, vel_lr_t, pos_hr_t, a_train = load_snapshot(
        data_dir, sim_train, snap_train, mesh_lr, mesh_hr, box_size
    )
    if pos_hr_t is None:
        raise RuntimeError("HR positions required for displacement target — check data_dir and mesh_hr.")
    logger.info(f"  a = {a_train:.4f}   N_lr = {pos_lr_t.shape[0]:,}   N_hr = {pos_hr_t.shape[0]:,}")

    # Deformation features
    feats_train, det_D_train = snapshot_features(
        pos_lr_t, neighbor_idx, mesh_lr, use_strain, use_invariants,
        ext_neighbor_idx, ext_offsets, env_pool_mode, shell_slices,
    )
    sc_frac_train = float(np.mean(det_D_train < 0))
    strain_mag    = float(jnp.mean(jnp.sqrt(jnp.sum(feats_train[:, :9] ** 2, axis=-1))))
    logger.info(f"  shell-crossing fraction: {sc_frac_train:.3%}   ||E||_F mean: {strain_mag:.4f}")

    # Displacement pair
    logger.info("Computing displacement pair …")
    psi_lr_tr, psi_hr_tr, delta_psi_tr = _disp_pair(pos_lr_t, pos_hr_t)

    disp_mag    = float(jnp.mean(jnp.sqrt(jnp.sum(delta_psi_tr ** 2, axis=-1))))
    abs_psi_mag = float(jnp.mean(jnp.sqrt(jnp.sum(psi_hr_tr ** 2, axis=-1))))
    logger.info(f"  |Ψ_HR| mean = {abs_psi_mag:.4e}   |ΔΨ| mean = {disp_mag:.4e}  "
                f"  ratio = {disp_mag / (abs_psi_mag + 1e-12):.3f}")

    target_train = delta_psi_tr
    if _apply_cnn is not None:
        logger.info("Subtracting CNN displacement correction from training target …")
        dpsi_cnn_tr  = _apply_cnn(pos_lr_t, vel_lr_t, jnp.array(a_train))
        target_train = delta_psi_tr - dpsi_cnn_tr
        cnn_mag      = float(jnp.mean(jnp.sqrt(jnp.sum(dpsi_cnn_tr  ** 2, axis=-1))))
        res_mag      = float(jnp.mean(jnp.sqrt(jnp.sum(target_train  ** 2, axis=-1))))
        logger.info(f"  |ΔΨ_CNN|={cnn_mag:.4e}   |ΔΨ_res|={res_mag:.4e}")

    vel_feat_tr = vel_lr_t if use_velocity else jnp.zeros_like(vel_lr_t)

    # ── Per-particle loss weights ─────────────────────────────────────────────
    if use_weighted_loss:
        weights_np = compute_sample_weights(
            pos_lr_t, det_D_train, mesh_lr,
            sc_boost      = loss_sc_boost,
            density_boost = loss_density_boost,
            density_gamma = loss_density_gamma,
        )
        logger.info(
            f"Sample weights: min={weights_np.min():.3f}  max={weights_np.max():.3f}  "
            f"w̄(SC)={weights_np[det_D_train < 0].mean() if (det_D_train < 0).any() else 1.0:.3f}"
        )
    else:
        weights_np = np.ones(pos_lr_t.shape[0], dtype=np.float32)

    weights_train = jnp.array(weights_np)

    # ── Init model ────────────────────────────────────────────────────────────
    rng    = jax.random.PRNGKey(seed)
    params = lag_model.init(rng, feats_train, vel_feat_tr, jnp.array(a_train))
    n_p    = sum(x.size for x in jax.tree_util.tree_leaves(params))
    logger.info(f"Parameters: {n_p:,}   feature dim: {int(feats_train.shape[1])}")

    wandb.log({
        "model/n_params":            n_p,
        "model/feat_dim":            int(feats_train.shape[1]),
        "model/n_shell":             n_shell,
        "data/sc_frac_train":        sc_frac_train,
        "data/strain_mag_train":     strain_mag,
        "data/disp_mag_target":      disp_mag,
        "data/abs_psi_mag":          abs_psi_mag,
        "loss_weight/sc_boost":      loss_sc_boost,
        "loss_weight/density_boost": loss_density_boost,
    }, step=0)

    # ── Optimiser ─────────────────────────────────────────────────────────────
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=lr_val,
        warmup_steps=warmup, decay_steps=n_steps, end_value=lr_val * 0.01,
    )
    optimizer  = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(lr_schedule, weight_decay=weight_decay),
    )
    opt_state  = optimizer.init(params)
    train_step = make_train_step(lag_model, optimizer)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_mse = float("inf")
    best_params  = None

    for step in range(1, n_steps + 1):
        params, opt_state, loss = train_step(
            params, opt_state,
            feats_train, vel_feat_tr, jnp.array(a_train),
            target_train, weights_train,
        )

        if step % log_every == 0 or step == 1:
            pred_tr  = jax.jit(lag_model.apply)(
                params, feats_train, vel_feat_tr, jnp.array(a_train)
            )
            log_dict = compute_disp_metrics(pred_tr, target_train, det_D_train, "train/")
            log_dict["train/loss"] = float(loss)
            log_dict["train/lr"]   = float(lr_schedule(step))

            # ── Validation ───────────────────────────────────────────────────
            val_mses = []
            for vsnap in snaps_val:
                vpos, vvel, vpos_hr, va = load_snapshot(
                    data_dir, sim_val, vsnap, mesh_lr, mesh_hr, box_size
                )
                vfeats, vdet_D = snapshot_features(
                    vpos, neighbor_idx, mesh_lr, use_strain, use_invariants,
                    ext_neighbor_idx, ext_offsets, env_pool_mode, shell_slices,
                )
                _, _, vdelta_psi = _disp_pair(vpos, vpos_hr)

                vtarget = vdelta_psi
                if _apply_cnn is not None:
                    vcnn = _apply_cnn(vpos, vvel, jnp.array(va))
                    vtarget = vdelta_psi - vcnn

                vvel_feat = vvel if use_velocity else jnp.zeros_like(vvel)
                vpred     = jax.jit(lag_model.apply)(
                    params, vfeats, vvel_feat, jnp.array(va)
                )
                vmet = compute_disp_metrics(vpred, vtarget, vdet_D, f"val/snap{vsnap}/")
                val_mses.append(vmet[f"val/snap{vsnap}/disp_mse"])
                log_dict.update(vmet)
                del vpos, vvel, vfeats, vtarget, vpred

            mean_val = float(np.mean(val_mses))
            log_dict["val/disp_mse_mean"] = mean_val
            wandb.log(log_dict, step=step)

            logger.info(
                f"step {step:5d}  loss={float(loss):.4e}"
                f"  R̄={log_dict['train/pearson_r_mean']:.3f}"
                f"  val_mse={mean_val:.4e}"
            )

            if mean_val < best_val_mse:
                best_val_mse = mean_val
                best_params  = jax.device_get(hk.data_structures.to_mutable_dict(params))

        if step % save_every == 0:
            ckpt = out_dir / f"params_step{step:05d}.pkl"
            with open(ckpt, "wb") as fh:
                pickle.dump(jax.device_get(hk.data_structures.to_mutable_dict(params)), fh)
            logger.info(f"  checkpoint → {ckpt}")

    # ── Save best & final ─────────────────────────────────────────────────────
    if best_params is not None:
        with open(out_dir / "best_params.pkl", "wb") as fh:
            pickle.dump(best_params, fh)

    with open(out_dir / "final_params.pkl", "wb") as fh:
        pickle.dump(jax.device_get(hk.data_structures.to_mutable_dict(params)), fh)

    wandb.log({"val/best_disp_mse": best_val_mse}, step=n_steps)
    logger.info(f"Done.  best_val_mse={best_val_mse:.4e}  output → {out_dir}")
    wandb.finish()


# ==============================================================================
# Entry point
# ==============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Lagrangian displacement super-resolution")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train(args.config)
