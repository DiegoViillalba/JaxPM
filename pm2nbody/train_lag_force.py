"""
train_lag_force.py — Single-snapshot Lagrangian force regression experiment.

Experiment design
-----------------
Given one simulation snapshot at scale factor a, we learn a per-particle
Lagrangian correction ΔF^Lag that minimises the residual force error:

  Standalone:   target = F_ref − F_PM
  Residual:     target = F_ref − F_PM − ΔF_CNN    (pre-trained CNN-WST supplied)

where:
  F_PM  = pm_forces(pos_lr_t, mesh_lr)                LR PM force
  F_ref = pm_forces(pos_lr→HR_scale, mesh_hr) × r     LR particles at HR resolution
          OR  pm_forces_at_lr_from_hr_density(pos_hr)  real HR particle density

The Lagrangian features z_i are the LOCAL STRAIN TENSOR:
  E = D − I,   D[i,α,β] = (x^α_i − x^α_{j,−β}) / Δq   (deformation gradient)

This is computed from the 6 axis-aligned Lagrangian neighbours (constant through
the whole simulation → precomputed once from particle count).

Physical motivation
-------------------
det(D) < 0 signals SHELL-CROSSING, invisible in the Eulerian density field.
The strain eigenvalues encode anisotropic collapse (filaments, sheets, halos).
Relative velocities within a patch identify kinematically distinct streams.

Usage
-----
  python train_lag_force.py --config configs/lag_force_mlp.yaml
  python train_lag_force.py --config configs/lag_force_cnn_wst.yaml
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
    compute_deformation_features,
    compute_deformation_tensor,
    make_lagrangian_corrector,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")


# ==============================================================================
# 1. Force pair computation
# ==============================================================================

def compute_force_pair(
    pos_lr_t: jnp.ndarray,
    mesh_lr: int,
    mesh_hr: int,
    pos_hr_t: jnp.ndarray = None,
) -> tuple:
    """
    Compute (F_PM, F_ref, ΔF) for one snapshot in LR mesh units.

    Physical force scales as (mesh / box_size) × F_mesh_units.
    To compare forces from different meshes at the same physical location:
        F_ref_lr_units = F_ref_hr_units × (mesh_hr / mesh_lr)

    Parameters
    ----------
    pos_lr_t  : [N_lr, 3]   LR positions in mesh_lr units
    mesh_lr   : int
    mesh_hr   : int
    pos_hr_t  : [N_hr, 3]   HR positions in mesh_lr units (optional).
                             None → LR particles used at HR resolution.

    Returns
    -------
    f_pm    : [N_lr, 3]   PM force at LR resolution
    f_ref   : [N_lr, 3]   Reference force at HR resolution (LR mesh units)
    delta_f : [N_lr, 3]   = f_ref − f_pm
    """
    r = float(mesh_hr) / float(mesh_lr)

    # ── LR force ──────────────────────────────────────────────────────────────
    pos_lr_mod = jnp.mod(pos_lr_t, mesh_lr)
    delta_lr   = get_delta(pos_lr_mod, (mesh_lr,) * 3)
    delta_lr_k = jnp.fft.rfftn(delta_lr)
    kvec_lr    = fftk((mesh_lr,) * 3)
    f_pm       = potential_kgrid_to_force_at_pos(delta_lr_k, pos_lr_mod, kvec_lr)

    # ── HR density source ─────────────────────────────────────────────────────
    # pos_hr_t is in mesh_lr units; scale to mesh_hr units for HR CIC painting.
    if pos_hr_t is None:
        hr_source = jnp.mod(pos_lr_t * r, mesh_hr)   # LR particles on HR mesh
    else:
        hr_source = jnp.mod(pos_hr_t * r, mesh_hr)   # actual HR particles

    # ── HR force at LR particle positions ─────────────────────────────────────
    delta_hr     = get_delta(hr_source, (mesh_hr,) * 3)
    delta_hr_k   = jnp.fft.rfftn(delta_hr)
    kvec_hr      = fftk((mesh_hr,) * 3)
    pos_lr_in_hr = jnp.mod(pos_lr_t * r, mesh_hr)    # LR pos in HR mesh units
    f_hr_raw     = potential_kgrid_to_force_at_pos(delta_hr_k, pos_lr_in_hr, kvec_hr)
    f_ref        = f_hr_raw * r    # convert HR gradient units → LR gradient units

    return f_pm, f_ref, f_ref - f_pm


def compute_cnn_force_correction(
    cnn_model,
    cnn_params,
    pos_lr_t: jnp.ndarray,
    vel_lr_t: jnp.ndarray,
    a: float,
    mesh_lr: int,
) -> jnp.ndarray:
    """
    CNN-WST force correction ΔF_CNN = −∇ΔΦ_CNN at all particles.

    Uses the same diagonal-Jacobian trick as pm.py: stop_gradient on positions
    in the transformer context, grad of scalar sum w.r.t. differentiable pos.

    Returns [N, 3] in LR mesh units.
    """
    pos_lr_mod   = jnp.mod(pos_lr_t, mesh_lr)
    delta        = get_delta(pos_lr_mod, (mesh_lr,) * 3)
    delta_lr_k   = jnp.fft.rfftn(delta)
    kvec_lr      = fftk((mesh_lr,) * 3)
    _, pm_pot    = potential_kgrid_to_force_at_pos(
        delta_lr_k, pos_lr_mod, kvec_lr, return_potential=True
    )
    grid_data = jnp.stack([pm_pot, delta], axis=-1)
    vel_sg    = jax.lax.stop_gradient(vel_lr_t)

    def phi_sum(pos):
        return jnp.sum(cnn_model.apply(cnn_params, grid_data, pos, a, vel_sg)[:, 0])

    return jax.grad(phi_sum)(pos_lr_t)   # [N, 3]


# ==============================================================================
# 2. Data loading
# ==============================================================================

def load_snapshot(
    data_dir: Path,
    sim_id: int,
    snap_idx: int,
    mesh_lr: int,
    mesh_hr: int,
    box_size: float,
    load_hr: bool = True,
) -> tuple:
    """
    Load one snapshot; returns positions and velocities in mesh_lr units.

    Returns (pos_lr, vel_lr, pos_hr_or_None, a)
    """
    scale = float(mesh_lr) / float(box_size)

    pos_lr = np.load(data_dir / f"pos_m{mesh_lr}_s{sim_id}.npy")[snap_idx]
    vel_lr = np.load(data_dir / f"vel_m{mesh_lr}_s{sim_id}.npy")[snap_idx]
    avals  = np.load(data_dir / "scale_factors.npy")
    a_val  = float(avals[snap_idx])

    pos_lr = jnp.array(pos_lr * scale, dtype=jnp.float32)
    vel_lr = jnp.array(vel_lr * scale, dtype=jnp.float32)

    pos_hr = None
    if load_hr:
        try:
            pos_hr_raw = np.load(data_dir / f"pos_m{mesh_hr}_s{sim_id}.npy")[snap_idx]
            pos_hr = jnp.array(pos_hr_raw * scale, dtype=jnp.float32)
        except FileNotFoundError:
            logger.warning(f"HR data not found for sim {sim_id}, using LR particles at HR resolution")

    return pos_lr, vel_lr, pos_hr, a_val


# ==============================================================================
# 3. Metrics
# ==============================================================================

def pearson_r(x: jnp.ndarray, y: jnp.ndarray) -> float:
    x_c = x - x.mean()
    y_c = y - y.mean()
    return float(
        jnp.sum(x_c * y_c) /
        (jnp.sqrt(jnp.sum(x_c ** 2) * jnp.sum(y_c ** 2)) + 1e-12)
    )


def compute_metrics(
    f_pred: jnp.ndarray,
    f_target: jnp.ndarray,
    det_D: np.ndarray,
    prefix: str = "",
) -> dict:
    """
    Force regression metrics.

    Reported per component (x/y/z), aggregate MSE, fractional MSE,
    Pearson R, and shell-crossing fraction from the Jacobian determinant.
    """
    err       = f_pred - f_target
    target_sq = jnp.sum(f_target ** 2, axis=-1)

    mse_full = float(jnp.mean(err ** 2))
    mse_x    = float(jnp.mean(err[:, 0] ** 2))
    mse_y    = float(jnp.mean(err[:, 1] ** 2))
    mse_z    = float(jnp.mean(err[:, 2] ** 2))
    frac_mse = float(jnp.mean(jnp.sum(err ** 2, axis=-1) / (target_sq + 1e-12)))
    r_x      = pearson_r(f_pred[:, 0], f_target[:, 0])
    r_y      = pearson_r(f_pred[:, 1], f_target[:, 1])
    r_z      = pearson_r(f_pred[:, 2], f_target[:, 2])
    sc_frac  = float(np.mean(det_D < 0.0))
    f_mag    = float(jnp.mean(jnp.sqrt(target_sq)))

    p = prefix
    return {
        f"{p}force_mse":        mse_full,
        f"{p}force_mse_x":      mse_x,
        f"{p}force_mse_y":      mse_y,
        f"{p}force_mse_z":      mse_z,
        f"{p}frac_mse":         frac_mse,
        f"{p}pearson_r_x":      r_x,
        f"{p}pearson_r_y":      r_y,
        f"{p}pearson_r_z":      r_z,
        f"{p}pearson_r_mean":   (r_x + r_y + r_z) / 3.0,
        f"{p}target_force_mag": f_mag,
        f"{p}shell_cross_frac": sc_frac,
    }


# ==============================================================================
# 4. Training helpers
# ==============================================================================

def compute_sample_weights(
    pos_lr_t: jnp.ndarray,
    det_D: np.ndarray,
    mesh_lr: int,
    sc_boost: float = 5.0,
    density_gamma: float = 0.5,
    density_boost: float = 2.0,
) -> np.ndarray:
    """
    Per-particle loss weights that emphasise shell-crossing and overdense regions.

    Physical motivation
    -------------------
    Shell-crossing (det D < 0) and high-density regions are rare but carry the
    strongest non-linear signal.  Uniform MSE under-weights them because their
    contribution to the average is diluted by the many low-density particles.

    Weight formula
    --------------
      w_i = [1 + (sc_boost − 1) · 𝟙[det D_i < 0]] · (1 + density_boost · max(δ_i, 0))^γ

      • sc_boost      — SC particles get up to sc_boost× more weight than normal.
      • density_gamma — exponent on overdensity; 0 = uniform, 1 = linear, 0.5 = sqrt.
      • density_boost — scale for overdensity term (set 0 to disable density weighting).

    Weights are normalised to mean=1 so the absolute MSE scale is preserved and
    loss curves from different runs remain directly comparable.

    Parameters
    ----------
    pos_lr_t      : [N, 3]   particle positions in mesh_lr units
    det_D         : [N]      Jacobian determinant (numpy, from snapshot_features)
    mesh_lr       : int
    sc_boost      : float    weight multiplier for det(D) < 0 particles  (default 5)
    density_gamma : float    exponent γ on the density term               (default 0.5)
    density_boost : float    scale α in (1 + α·δ)^γ                      (default 2.0)

    Returns
    -------
    weights : [N] float32 numpy array, mean = 1.0
    """
    N = len(det_D)
    w = np.ones(N, dtype=np.float32)

    # ── Shell-crossing boost ─────────────────────────────────────────────────
    if sc_boost > 1.0:
        sc_mask = det_D < 0
        w[sc_mask] *= float(sc_boost)

    # ── Density boost ────────────────────────────────────────────────────────
    # Paint particles to get the overdensity field, then nearest-neighbour read.
    # Weights are not back-propagated, so NN is fine and avoids extra CIC imports.
    if density_boost > 0.0 and density_gamma > 0.0:
        pos_mod      = np.asarray(jax.device_get(jnp.mod(pos_lr_t, mesh_lr)))
        delta_field  = np.asarray(jax.device_get(
            get_delta(jnp.array(pos_mod, dtype=jnp.float32), (mesh_lr,) * 3)
        ))                                    # [M, M, M]  overdensity δ = ρ/ρ̄ − 1
        idx = np.round(pos_mod).astype(int) % mesh_lr   # [N, 3]  nearest cell
        delta_part   = delta_field[idx[:, 0], idx[:, 1], idx[:, 2]]  # [N]
        delta_pos    = np.maximum(delta_part, 0.0)       # only upweight overdense
        w *= (1.0 + float(density_boost) * delta_pos) ** float(density_gamma)

    # ── Normalise to mean=1 ──────────────────────────────────────────────────
    w /= w.mean()
    return w.astype(np.float32)


def make_train_step(model, optimizer):
    """
    Return a jit-compiled train step that supports per-particle loss weights.

    Signature: (params, opt_state, feats, vel, a, target, weights) → (params, opt_state, loss)

    weights : [N] float32, mean=1.  Pass jnp.ones(N) for uniform loss.
    """

    @jax.jit
    def step(params, opt_state, feats, vel, a, target, weights):
        def loss_fn(p):
            pred          = model.apply(p, feats, vel, a)           # [N, 3]
            per_particle  = jnp.sum((pred - target) ** 2, axis=-1)  # [N]
            return jnp.mean(per_particle * weights)                  # weighted MSE

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_state, loss

    return step


# ==============================================================================
# 5. Feature + det_D helper (avoids repeated recompute)
# ==============================================================================

def snapshot_features(
    pos_t,
    neighbor_idx,
    mesh_lr,
    use_strain,
    use_invariants,
    ext_neighbor_idx=None,
    ext_offsets=None,
    pool_mode: str = "mean_var",
    shell_slices: tuple = (),
):
    """
    Compute (feats [N, D], det_D [N]) for a single snapshot.

    Extended neighbourhood arguments (ext_neighbor_idx, ext_offsets, pool_mode,
    shell_slices) are forwarded to compute_deformation_features when provided.
    They are used by the displacement SR pipeline (train_lag_disp.py) to add
    non-linear environment residuals as additional features.
    """
    # static_argnums covers: mesh_lr(2), use_strain(3), use_invariants(4),
    # pool_mode(7), shell_slices(8)
    feats = jax.jit(
        compute_deformation_features, static_argnums=(2, 3, 4, 7, 8)
    )(
        pos_t, neighbor_idx, mesh_lr, use_strain, use_invariants,
        ext_neighbor_idx, ext_offsets, pool_mode, shell_slices,
    )

    neg_idx = neighbor_idx[:, [1, 3, 5]]
    _, det_D = jax.jit(
        lambda p: compute_deformation_tensor(p, neg_idx, mesh_lr)
    )(pos_t)
    det_D = np.asarray(jax.device_get(det_D))
    return feats, det_D


# ==============================================================================
# 6. Main training loop
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
        project=wb_cfg.get("project", "pm2nbody_lag_force"),
        name=getattr(exp_cfg, "name", None),
        tags=wb_cfg.get("tags", []),
        config=cfg,
    )

    out_dir = Path(getattr(exp_cfg, "output_dir", "runs/lag_force")) / wandb.run.name
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, out_dir / "config.yaml")

    # ── Config extraction ─────────────────────────────────────────────────────
    mesh_lr       = int(data_cfg.mesh_lr)
    mesh_hr       = int(data_cfg.mesh_hr)
    box_size      = float(data_cfg.box_size)
    n_part        = int(data_cfg.n_particles)
    data_dir      = Path(data_cfg.data_dir)
    snap_train    = int(data_cfg.snap_train)
    sim_train     = int(getattr(data_cfg, "sim_id_train",  0))
    sim_val       = int(getattr(data_cfg, "sim_id_val",    1))
    snaps_val     = list(getattr(data_cfg, "snaps_val",    [snap_train]))

    use_strain     = bool(getattr(model_cfg, "use_strain",     True))
    use_invariants = bool(getattr(model_cfg, "use_invariants", False))
    use_velocity   = bool(getattr(model_cfg, "use_velocity",   True))
    output_mode    = getattr(model_cfg, "output_mode", "force")
    output_dim     = 1 if output_mode == "potential" else 3

    n_steps      = int(getattr(train_cfg, "n_steps",      500))
    lr_val       = float(getattr(train_cfg, "lr",          3e-4))
    weight_decay = float(getattr(train_cfg, "weight_decay", 1e-4))
    warmup       = int(getattr(train_cfg, "warmup_steps",   50))
    log_every    = int(getattr(train_cfg, "log_every",      25))
    save_every   = int(getattr(train_cfg, "save_every",    100))
    seed         = int(getattr(train_cfg, "seed",            0))

    # ── Loss weighting ────────────────────────────────────────────────────────
    loss_sc_boost       = float(getattr(train_cfg, "loss_sc_boost",       1.0))
    loss_density_boost  = float(getattr(train_cfg, "loss_density_boost",  0.0))
    loss_density_gamma  = float(getattr(train_cfg, "loss_density_gamma",  0.5))
    use_weighted_loss   = (loss_sc_boost > 1.0) or (loss_density_boost > 0.0)
    logger.info(
        f"Loss weighting: sc_boost={loss_sc_boost}  "
        f"density_boost={loss_density_boost}  density_gamma={loss_density_gamma}  "
        f"({'active' if use_weighted_loss else 'uniform — set loss_sc_boost>1 or loss_density_boost>0 to enable'})"
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    lag_model = make_lagrangian_corrector(
        hidden_dim = int(getattr(model_cfg, "hidden_dim", 64)),
        n_layers   = int(getattr(model_cfg, "n_layers",   3)),
        output_dim = output_dim,
    )
    logger.info(
        f"Model: hidden_dim={int(getattr(model_cfg, 'hidden_dim', 64))}  "
        f"n_layers={int(getattr(model_cfg, 'n_layers', 3))}  "
        f"output_dim={output_dim}"
    )

    # ── Optional CNN-WST residual mode ────────────────────────────────────────
    cnn_model, cnn_params = None, None
    if getattr(model_cfg, "cnn_checkpoint", None):
        logger.info(f"Loading CNN-WST: {model_cfg.cnn_checkpoint}")
        try:
            import pm2nbody.eval_utils as eu
            ri = eu.load_run(Path(model_cfg.cnn_checkpoint))
            cnn_model, cnn_params = ri["model"], ri["params"]
        except Exception as e:
            logger.error(f"Could not load CNN checkpoint: {e}")

    # ── Pre-compile force helpers (capture static values in closure) ─────────
    # cnn_model is a Haiku NamedTuple (not a JAX array), so it must be captured
    # as a Python closure — NOT passed as a jit argument.
    # pos_hr_t may be None, which is also not a valid JAX array; two jitted
    # variants handle the with-HR and without-HR cases separately.
    if cnn_model is not None:
        _apply_cnn_force = jax.jit(
            lambda pos, vel, a: compute_cnn_force_correction(
                cnn_model, cnn_params, pos, vel, a, mesh_lr
            )
        )
    else:
        _apply_cnn_force = None

    _force_pair_lr_only = jax.jit(
        lambda pos: compute_force_pair(pos, mesh_lr, mesh_hr, None)
    )
    _force_pair_with_hr = jax.jit(
        lambda pos, pos_hr: compute_force_pair(pos, mesh_lr, mesh_hr, pos_hr)
    )

    def _get_force_pair(pos, pos_hr):
        """Dispatch to the right jit-compiled variant depending on HR availability."""
        return _force_pair_lr_only(pos) if pos_hr is None else _force_pair_with_hr(pos, pos_hr)

    # ── Lagrangian structure (precomputed once) ───────────────────────────────
    logger.info(f"Precomputing Lagrangian neighbours  n_part³ = {n_part}³ = {n_part**3:,}")
    neighbor_idx = get_axis_neighbor_indices(n_part)   # [N, 6]

    # ── Training snapshot ─────────────────────────────────────────────────────
    logger.info(f"Loading training snapshot {snap_train}  sim {sim_train}")
    pos_lr_t, vel_lr_t, pos_hr_t, a_train = load_snapshot(
        data_dir, sim_train, snap_train, mesh_lr, mesh_hr, box_size
    )
    logger.info(f"  a = {a_train:.4f}   N_lr = {pos_lr_t.shape[0]:,}")

    # Deformation features
    feats_train, det_D_train = snapshot_features(
        pos_lr_t, neighbor_idx, mesh_lr, use_strain, use_invariants
    )
    sc_frac_train = float(np.mean(det_D_train < 0))
    strain_mag    = float(jnp.mean(jnp.sqrt(jnp.sum(feats_train[:, :9] ** 2, axis=-1))))
    logger.info(f"  shell-crossing fraction: {sc_frac_train:.3%}   ||E||_F mean: {strain_mag:.4f}")

    # Force pair
    logger.info("Computing force pairs …")
    f_pm_tr, f_ref_tr, delta_f_tr = _get_force_pair(pos_lr_t, pos_hr_t)

    target_train = delta_f_tr
    if _apply_cnn_force is not None:
        logger.info("Subtracting CNN-WST correction from training target …")
        f_cnn_tr = _apply_cnn_force(pos_lr_t, vel_lr_t, jnp.array(a_train))
        target_train = delta_f_tr - f_cnn_tr
        cnn_mag = float(jnp.mean(jnp.sqrt(jnp.sum(f_cnn_tr ** 2, axis=-1))))
        raw_mag = float(jnp.mean(jnp.sqrt(jnp.sum(delta_f_tr ** 2, axis=-1))))
        res_mag = float(jnp.mean(jnp.sqrt(jnp.sum(target_train ** 2, axis=-1))))
        logger.info(
            f"  |ΔF_raw|={raw_mag:.4e}  |ΔF_cnn|={cnn_mag:.4e}  "
            f"|ΔF_residual|={res_mag:.4e}"
        )

    vel_feat_tr = vel_lr_t if use_velocity else jnp.zeros_like(vel_lr_t)

    # ── Per-particle loss weights ─────────────────────────────────────────────
    # Computed once on the training snapshot.  Validation always uses uniform
    # weights (val MSE stays a clean, unbiased metric).
    if use_weighted_loss:
        logger.info(
            f"Computing sample weights  (sc_boost={loss_sc_boost}, "
            f"density_boost={loss_density_boost}, gamma={loss_density_gamma}) …"
        )
        weights_train_np = compute_sample_weights(
            pos_lr_t, det_D_train, mesh_lr,
            sc_boost       = loss_sc_boost,
            density_boost  = loss_density_boost,
            density_gamma  = loss_density_gamma,
        )
        sc_weight_mean   = float(weights_train_np[det_D_train < 0].mean()) if (det_D_train < 0).any() else 1.0
        nrm_weight_mean  = float(weights_train_np[det_D_train >= 0].mean())
        logger.info(
            f"  weight stats: mean=1.00  min={weights_train_np.min():.3f}  "
            f"max={weights_train_np.max():.3f}  "
            f"w̄(SC)={sc_weight_mean:.3f}  w̄(normal)={nrm_weight_mean:.3f}"
        )
    else:
        weights_train_np = np.ones(pos_lr_t.shape[0], dtype=np.float32)
        sc_weight_mean, nrm_weight_mean = 1.0, 1.0

    weights_train = jnp.array(weights_train_np)   # move to device once

    # ── Initialise ────────────────────────────────────────────────────────────
    rng    = jax.random.PRNGKey(seed)
    params = lag_model.init(rng, feats_train, vel_feat_tr, jnp.array(a_train))
    n_p    = sum(x.size for x in jax.tree_util.tree_leaves(params))
    logger.info(f"Parameters: {n_p:,}")
    wandb.log({
        "model/n_params":              n_p,
        "data/sc_frac_train":          sc_frac_train,
        "data/strain_mag_train":       strain_mag,
        "loss_weight/sc_boost":        loss_sc_boost,
        "loss_weight/density_boost":   loss_density_boost,
        "loss_weight/density_gamma":   loss_density_gamma,
        "loss_weight/w_mean_sc":       sc_weight_mean,
        "loss_weight/w_mean_normal":   nrm_weight_mean,
        "loss_weight/w_max":           float(weights_train_np.max()),
    }, step=0)

    # ── Optimiser ─────────────────────────────────────────────────────────────
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=lr_val,
        warmup_steps=warmup,
        decay_steps=n_steps,
        end_value=lr_val * 0.01,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(lr_schedule, weight_decay=weight_decay),
    )
    opt_state  = optimizer.init(params)
    train_step = make_train_step(lag_model, optimizer)

    # ── Loop ──────────────────────────────────────────────────────────────────
    best_val_mse = float("inf")
    best_params  = None

    for step in range(1, n_steps + 1):
        params, opt_state, loss = train_step(
            params, opt_state,
            feats_train, vel_feat_tr, jnp.array(a_train), target_train,
            weights_train,
        )

        if step % log_every == 0 or step == 1:
            f_pred_tr = jax.jit(lag_model.apply)(
                params, feats_train, vel_feat_tr, jnp.array(a_train)
            )
            log_dict = compute_metrics(f_pred_tr, target_train, det_D_train, "train/")
            log_dict["train/loss"] = float(loss)
            log_dict["train/lr"]   = float(lr_schedule(step))

            # ── Validation ────────────────────────────────────────────────────
            val_mses = []
            for vsnap in snaps_val:
                vpos, vvel, vpos_hr, va = load_snapshot(
                    data_dir, sim_val, vsnap, mesh_lr, mesh_hr, box_size
                )
                vfeats, vdet_D = snapshot_features(
                    vpos, neighbor_idx, mesh_lr, use_strain, use_invariants
                )
                _, _, vdelta_f = _get_force_pair(vpos, vpos_hr)

                vtarget = vdelta_f
                if _apply_cnn_force is not None:
                    vf_cnn  = _apply_cnn_force(vpos, vvel, jnp.array(va))
                    vtarget = vdelta_f - vf_cnn

                vvel_feat = vvel if use_velocity else jnp.zeros_like(vvel)
                vf_pred   = jax.jit(lag_model.apply)(
                    params, vfeats, vvel_feat, jnp.array(va)
                )
                vmet = compute_metrics(vf_pred, vtarget, vdet_D, f"val/snap{vsnap}/")
                val_mses.append(vmet[f"val/snap{vsnap}/force_mse"])
                log_dict.update(vmet)

                del vpos, vvel, vfeats, vtarget, vf_pred

            mean_val = float(np.mean(val_mses))
            log_dict["val/force_mse_mean"] = mean_val
            wandb.log(log_dict, step=step)

            logger.info(
                f"step {step:5d}  loss={float(loss):.4e}"
                f"  R̄={log_dict['train/pearson_r_mean']:.3f}"
                f"  val_mse={mean_val:.4e}"
            )

            if mean_val < best_val_mse:
                best_val_mse = mean_val
                best_params  = jax.device_get(hk.data_structures.to_mutable_dict(params))
                with open(out_dir / "best_params.pkl", "wb") as fh:
                    pickle.dump(best_params, fh)
                wandb.summary.update({"best_val_mse": best_val_mse, "best_step": step})

        if step % save_every == 0:
            p_path = out_dir / f"params_{step:05d}.pkl"
            with open(p_path, "wb") as fh:
                pickle.dump(jax.device_get(hk.data_structures.to_mutable_dict(params)), fh)
            logger.info(f"  ✓ checkpoint → {p_path.name}")

    # ── Final save ────────────────────────────────────────────────────────────
    with open(out_dir / "final_params.pkl", "wb") as fh:
        pickle.dump(jax.device_get(hk.data_structures.to_mutable_dict(params)), fh)

    wandb.finish()
    logger.info(f"Done.  best_val_mse={best_val_mse:.4e}  → {out_dir}")
    return best_params


# ==============================================================================
# Entry point
# ==============================================================================

if __name__ == "__main__":
    from absl import flags, app

    FLAGS = flags.FLAGS
    flags.DEFINE_string("config", None, "Path to YAML config")
    flags.mark_flag_as_required("config")

    def main(_):
        train(FLAGS.config)

    app.run(main)