"""
train_subregion_force.py — CNN+MLP sub-region force prediction.

Goal
----
Given a global coarse (LR) simulation and a spatial sub-region of HR particles,
predict the PM force correction for HR particles within that sub-region:

    F_total = F_LR@HR + ΔF_CNN + ΔF_MLP

Where:
  F_LR@HR  = PM force from LR density evaluated at HR particle positions  (baseline)
  ΔF_CNN   = force correction from CNN-WST on the global LR density grid
  ΔF_MLP   = per-particle correction from Lagrangian MLP at HR resolution
  target   = ΔF_mass = F_HR − F_LR@HR  (full mass-resolution correction)

Key differences from train_lag_massres.py:
  • Force pair evaluated at ALL HR positions (no stride-subsampling)
  • Lagrangian features computed at HR resolution (mesh_hr grid, mesh_hr units)
  • Training is patch-based: random Lagrangian sub-cubes at each step
  • CNN evaluates at HR patch positions (not LR positions)

Two-stage support:
  Stage 1 (optional): train_cnn_massres.py    → CNN-WST baseline
  Stage 2 (this):     train_subregion_force.py → MLP residual on HR patches

Data:
  Output of generate_data_subregion.py:
    pos_m{lr}_s{n}.npy           [n_snaps, mesh_lr³, 3]   LR positions (Mpc/h)
    pos_m{hr}_s{n}.npy           [n_snaps, mesh_hr³, 3]   HR positions (Mpc/h)
    vel_m{lr}_s{n}.npy           [n_snaps, mesh_lr³, 3]   LR velocities
    delta_f_m{hr}_s{n}.npy       [n_snaps, mesh_hr³, 3]   ΔF target (mesh_lr units)
    scale_factors.npy            [n_snaps]

Usage:
  python train_subregion_force.py --config configs/subregion_force.yaml
"""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.85"

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

from scipy.stats import pearsonr

from jaxpm.lagrangian import (
    get_axis_neighbor_indices,
    get_shell_neighbor_indices,
    make_lagrangian_corrector,
)
from jaxpm.pm import get_delta

# ── Shared helpers ─────────────────────────────────────────────────────────────
from train_lag_force import (
    snapshot_features,
    make_train_step,
    compute_sample_weights,
    compute_metrics,
)
from train_cnn_massres import (
    build_grid_data,
    compute_cnn_pred,
    build_cnn_model,
)
from train_lag_massres import (
    compute_subregion_force_pair,
    _load_cnn_massres_checkpoint,
    compute_cnn_massres_correction,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")


# ==============================================================================
# 1. Data loading
# ==============================================================================

def load_subregion_snapshot(
    data_dir: Path,
    sim_id: int,
    snap_idx: int,
    mesh_lr: int,
    mesh_hr: int,
    box_size: float,
    load_velocities: bool = True,
) -> dict:
    """
    Load one snapshot for sub-region training.

    Returns a dict with:
      pos_lr   [mesh_lr³, 3]  LR positions in mesh_lr units
      vel_lr   [mesh_lr³, 3]  LR velocities in mesh_lr/a units
      pos_hr   [mesh_hr³, 3]  HR positions in mesh_lr units
      vel_hr   [mesh_hr³, 3]  HR velocities (only if load_velocities=True)
      delta_f  [mesh_hr³, 3]  ΔF = F_HR − F_LR@HR  in mesh_lr units
      a        float           scale factor
    """
    data_dir = Path(data_dir)
    scale    = float(mesh_lr) / float(box_size)   # Mpc/h → mesh_lr units
    avals    = np.load(data_dir / "scale_factors.npy")
    a        = float(avals[snap_idx])

    pos_lr = np.load(data_dir / f"pos_m{mesh_lr}_s{sim_id}.npy")[snap_idx] * scale
    pos_hr = np.load(data_dir / f"pos_m{mesh_hr}_s{sim_id}.npy")[snap_idx] * scale

    # delta_f is saved in mesh_lr units (no scaling needed)
    delta_f_path = data_dir / f"delta_f_m{mesh_hr}_s{sim_id}.npy"
    if not delta_f_path.exists():
        raise FileNotFoundError(
            f"Force file not found: {delta_f_path}\n"
            "Run generate_data_subregion.py first to pre-compute force fields."
        )
    delta_f = np.load(delta_f_path)[snap_idx]

    out = dict(
        pos_lr  = jnp.array(pos_lr,  dtype=jnp.float32),
        pos_hr  = jnp.array(pos_hr,  dtype=jnp.float32),
        delta_f = jnp.array(delta_f, dtype=jnp.float32),
        a       = a,
    )

    vel_lr_path = data_dir / f"vel_m{mesh_lr}_s{sim_id}.npy"
    if vel_lr_path.exists():
        out["vel_lr"] = jnp.array(
            np.load(vel_lr_path)[snap_idx] * scale, dtype=jnp.float32
        )
    else:
        out["vel_lr"] = jnp.zeros_like(out["pos_lr"])

    if load_velocities:
        vel_hr_path = data_dir / f"vel_m{mesh_hr}_s{sim_id}.npy"
        if vel_hr_path.exists():
            out["vel_hr"] = jnp.array(
                np.load(vel_hr_path)[snap_idx] * scale, dtype=jnp.float32
            )
        else:
            out["vel_hr"] = jnp.zeros_like(out["pos_hr"])

    return out


# ==============================================================================
# 2. HR Lagrangian features
# ==============================================================================

def compute_hr_features(
    pos_hr: jnp.ndarray,
    neighbor_idx_hr,
    mesh_hr: int,
    mesh_lr: int,
    use_strain: bool,
    use_invariants: bool,
    ext_neighbor_idx=None,
    ext_offsets=None,
    env_pool_mode: str = "mean_var",
    shell_slices=(),
) -> tuple:
    """
    Compute Lagrangian features for HR particles.

    Positions are converted from mesh_lr units to mesh_hr units (pos_hr * r)
    before feature computation. This ensures the deformation tensor has the
    same dimensionless scale as LR features computed with mesh_lr units.

    Parameters
    ----------
    pos_hr           : [mesh_hr³, 3] HR positions in mesh_lr units (from load_snapshot)
    neighbor_idx_hr  : [mesh_hr³, 6] axis-neighbour indices for the HR grid
    mesh_hr, mesh_lr : mesh sizes

    Returns
    -------
    feats    : [mesh_hr³, feat_dim]
    det_D    : [mesh_hr³]  determinant of deformation tensor (neg = shell-crossing)
    """
    r = float(mesh_hr) / float(mesh_lr)
    pos_hr_mh = pos_hr * r   # convert to mesh_hr units; Lagrangian spacing = 1.0 in these units
    return snapshot_features(
        pos_hr_mh, neighbor_idx_hr, mesh_hr,
        use_strain, use_invariants,
        ext_neighbor_idx, ext_offsets, env_pool_mode, shell_slices,
    )


# ==============================================================================
# 3. Patch sampling
# ==============================================================================

def sample_lag_patch(
    rng_key,
    mesh_hr: int,
    patch_n: int,
) -> np.ndarray:
    """
    Sample a random contiguous Lagrangian cube of size patch_n³.

    Returns flat indices [patch_n³] into the HR particle array (C-order, indexing='ij').
    The patch wraps periodically at the box boundary, consistent with the periodic
    Lagrangian grid used by make_lagrangian_grid.

    Flat index convention: m = ix * mesh_hr² + iy * mesh_hr + iz  (indexing='ij')
    """
    if isinstance(rng_key, int):
        rng = np.random.default_rng(rng_key)
        ox, oy, oz = rng.integers(0, mesh_hr, size=3)
    else:
        k0, k1, k2 = jax.random.split(rng_key, 3)
        ox = int(jax.random.randint(k0, (), 0, mesh_hr))
        oy = int(jax.random.randint(k1, (), 0, mesh_hr))
        oz = int(jax.random.randint(k2, (), 0, mesh_hr))

    ix = (np.arange(patch_n) + ox) % mesh_hr   # [patch_n]
    iy = (np.arange(patch_n) + oy) % mesh_hr
    iz = (np.arange(patch_n) + oz) % mesh_hr

    IXX, IYY, IZZ = np.meshgrid(ix, iy, iz, indexing='ij')   # [patch_n³]
    flat = (IXX.ravel() * mesh_hr * mesh_hr
            + IYY.ravel() * mesh_hr
            + IZZ.ravel()).astype(np.int32)
    return flat


def sample_lag_patch_with_border(
    rng_key,
    mesh_hr: int,
    patch_n: int,
    border: int,
) -> tuple:
    """
    Sample a patch of size (patch_n + 2*border)³ in Lagrangian space.

    Returns:
      inner_idx : [patch_n³]              flat indices — loss is computed here
      outer_idx : [(patch_n+2*border)³]   flat indices — features are computed here
                  (includes inner + border region for accurate Lagrangian features)
    """
    # Outer patch (for feature computation)
    outer_patch_n = patch_n + 2 * border
    outer_idx = sample_lag_patch(rng_key, mesh_hr, outer_patch_n)

    # Inner patch starts at offset border within the outer patch
    # Re-derive the origin of the outer patch to compute the inner patch
    if isinstance(rng_key, int):
        rng = np.random.default_rng(rng_key)
        ox, oy, oz = rng.integers(0, mesh_hr, size=3)
    else:
        k0, k1, k2 = jax.random.split(rng_key, 3)
        ox = int(jax.random.randint(k0, (), 0, mesh_hr))
        oy = int(jax.random.randint(k1, (), 0, mesh_hr))
        oz = int(jax.random.randint(k2, (), 0, mesh_hr))

    # Inner patch starts at (ox + border, oy + border, oz + border)
    ix = (np.arange(patch_n) + ox + border) % mesh_hr
    iy = (np.arange(patch_n) + oy + border) % mesh_hr
    iz = (np.arange(patch_n) + oz + border) % mesh_hr
    IXX, IYY, IZZ = np.meshgrid(ix, iy, iz, indexing='ij')
    inner_idx = (IXX.ravel() * mesh_hr * mesh_hr
                 + IYY.ravel() * mesh_hr
                 + IZZ.ravel()).astype(np.int32)

    return inner_idx, outer_idx


# ==============================================================================
# 4. CNN correction at HR patch positions
# ==============================================================================

def compute_cnn_at_hr_patch(
    cnn_model,
    cnn_params,
    pos_lr: jnp.ndarray,
    pos_hr_patch: jnp.ndarray,
    mesh_lr: int,
    a: float,
    use_pm_potential: bool = True,
) -> jnp.ndarray:
    """
    Compute CNN force correction at HR patch positions.

    The CNN grid is built from the LR density (global context). The gradient of
    the CNN scalar potential is evaluated at HR patch positions (in mesh_lr units),
    which lie within the same [0, mesh_lr) domain as the LR grid.

    Parameters
    ----------
    pos_lr       : [mesh_lr³, 3]  LR positions in mesh_lr units
    pos_hr_patch : [N_patch, 3]   HR patch positions in mesh_lr units
    mesh_lr      : LR mesh size
    a            : scale factor

    Returns
    -------
    [N_patch, 3]  CNN force correction in mesh_lr units
    """
    pos_lr_mod      = jnp.mod(pos_lr,       mesh_lr)
    pos_hr_patch_mod = jnp.mod(pos_hr_patch, mesh_lr)

    # LR density grid for CNN
    grid_data = build_grid_data(pos_lr_mod, mesh_lr, use_pm_potential)

    # CNN gradient at HR patch positions (CIC sub-cell interpolation from LR grid)
    return compute_cnn_pred(cnn_model, cnn_params, grid_data, pos_hr_patch_mod, a)


# ==============================================================================
# 5. Validation
# ==============================================================================

def val_metrics_subregion(
    lag_model,
    params: hk.Params,
    cnn_model,
    cnn_params,
    data_dir: Path,
    sim_id: int,
    snap_ids: list,
    mesh_lr: int,
    mesh_hr: int,
    box_size: float,
    neighbor_idx_hr,
    use_strain: bool,
    use_invariants: bool,
    use_velocity: bool,
    use_pm_potential: bool,
    n_val_patches: int = 8,
    patch_n: int = 32,
    ext_neighbor_idx=None,
    ext_offsets=None,
    env_pool_mode: str = "mean_var",
    shell_slices=(),
    prefix: str = "val/",
) -> tuple:
    """
    Evaluate the full sub-region pipeline on held-out snapshots.

    For each snapshot, samples n_val_patches random patches and computes:
      mse_lr       : MSE( F_LR@HR, F_HR )       baseline
      mse_corrected: MSE( F_LR@HR + ΔF_pred, F_HR )
      pearson_r    : R between prediction and target
    """
    r = float(mesh_hr) / float(mesh_lr)
    log = {}
    mse_improvements = []

    for snap_id in snap_ids:
        snap = load_subregion_snapshot(
            data_dir, sim_id, snap_id, mesh_lr, mesh_hr, box_size
        )
        pos_lr   = snap["pos_lr"]
        pos_hr   = snap["pos_hr"]
        delta_f  = snap["delta_f"]
        a_val    = snap["a"]
        vel_hr   = snap.get("vel_hr", jnp.zeros_like(pos_hr))

        # HR features for full snapshot
        feats_hr, det_D_hr = compute_hr_features(
            pos_hr, neighbor_idx_hr, mesh_hr, mesh_lr,
            use_strain, use_invariants,
            ext_neighbor_idx, ext_offsets, env_pool_mode, shell_slices,
        )

        # Load LR forces at HR positions (from delta_f and HR/LR components if available)
        # For MSE improvement we need F_LR@HR and F_HR
        # f_lr_at_hr + delta_f = f_hr  ⟹  f_lr_at_hr = f_hr - delta_f
        # We'll compute F_LR@HR from the force pair directly
        f_lr_at_hr, f_hr, _ = compute_subregion_force_pair(
            pos_lr, pos_hr, mesh_lr, mesh_hr,
        )
        f_lr_at_hr_np = np.asarray(jax.device_get(f_lr_at_hr))
        f_hr_np       = np.asarray(jax.device_get(f_hr))

        # CNN correction for full snapshot (expensive but done once per snap)
        if cnn_model is not None:
            pos_lr_mod = jnp.mod(pos_lr, mesh_lr)
            grid_data  = build_grid_data(pos_lr_mod, mesh_lr, use_pm_potential)
            cnn_full   = compute_cnn_pred(cnn_model, cnn_params, grid_data,
                                          jnp.mod(pos_hr, mesh_lr), a_val)
            cnn_full_np = np.asarray(jax.device_get(cnn_full))
        else:
            cnn_full_np = np.zeros_like(f_lr_at_hr_np)

        # MLP for full snapshot
        vel_feat = vel_hr if use_velocity else jnp.zeros_like(vel_hr)
        mlp_pred_np = np.asarray(jax.device_get(
            jax.jit(lag_model.apply)(params, feats_hr, vel_feat, jnp.array(a_val))
        ))

        # Total corrected force
        f_corrected = f_lr_at_hr_np + cnn_full_np + mlp_pred_np

        mse_lr  = float(np.mean((f_lr_at_hr_np - f_hr_np) ** 2))
        mse_cor = float(np.mean((f_corrected   - f_hr_np) ** 2))
        improv  = 1.0 - mse_cor / mse_lr
        mse_improvements.append(improv)

        # Target for MLP scatter (residual after CNN)
        delta_f_np = np.asarray(jax.device_get(delta_f))
        target_np  = delta_f_np - cnn_full_np
        r_vals     = [pearsonr(mlp_pred_np[:, c], target_np[:, c])[0] for c in range(3)]

        log[f"{prefix}snap{snap_id}/mse_lr"]        = mse_lr
        log[f"{prefix}snap{snap_id}/mse_corrected"] = mse_cor
        log[f"{prefix}snap{snap_id}/improvement"]   = improv
        log[f"{prefix}snap{snap_id}/pearson_r_mean"]= float(np.mean(r_vals))

    log[f"{prefix}improvement_mean"] = float(np.mean(mse_improvements))
    return log, float(np.mean(mse_improvements))


# ==============================================================================
# 6. Main training loop
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
        project = wb_cfg.get("project", "pm2nbody_subregion"),
        name    = getattr(exp_cfg, "name", None),
        tags    = wb_cfg.get("tags", []),
        config  = cfg,
    )
    out_dir = Path(getattr(exp_cfg, "output_dir", "runs/subregion_force")) / wandb.run.name
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, out_dir / "config.yaml")

    # ── Config ────────────────────────────────────────────────────────────────
    mesh_lr    = int(data_cfg.mesh_lr)
    mesh_hr    = int(data_cfg.mesh_hr)
    box_size   = float(data_cfg.box_size)
    data_dir   = Path(data_cfg.data_dir)
    sim_train  = int(getattr(data_cfg, "sim_id_train", 0))
    sim_val    = int(getattr(data_cfg, "sim_id_val",   1))
    snaps_val  = list(getattr(data_cfg, "snaps_val",   [5]))
    snap_train = int(getattr(data_cfg, "snap_train",   5))
    r          = float(mesh_hr) / float(mesh_lr)

    # Model config
    use_strain     = bool(getattr(model_cfg, "use_strain",     True))
    use_invariants = bool(getattr(model_cfg, "use_invariants", False))
    use_velocity   = bool(getattr(model_cfg, "use_velocity",   False))
    n_shell        = int(getattr(model_cfg,  "n_shell",        0))
    env_pool_mode  = str(getattr(model_cfg,  "env_pool_mode",  "mean_var"))
    hidden_dim     = int(getattr(model_cfg,  "hidden_dim",     64))
    n_layers       = int(getattr(model_cfg,  "n_layers",       3))
    use_pm_pot     = bool(getattr(model_cfg, "use_pm_potential", True))

    # Training config
    n_steps          = int(getattr(train_cfg, "n_steps",            5000))
    lr_val           = float(getattr(train_cfg, "lr",               3e-4))
    weight_decay     = float(getattr(train_cfg, "weight_decay",     1e-4))
    warmup           = int(getattr(train_cfg,   "warmup_steps",      100))
    log_every        = int(getattr(train_cfg,   "log_every",         100))
    save_every       = int(getattr(train_cfg,   "save_every",        500))
    seed             = int(getattr(train_cfg,   "seed",                0))
    patch_n          = int(getattr(train_cfg,   "patch_n",            32))
    patches_per_snap = int(getattr(train_cfg,   "patches_per_snap",   16))
    loss_sc_boost    = float(getattr(train_cfg, "loss_sc_boost",      1.0))
    loss_dens_boost  = float(getattr(train_cfg, "loss_density_boost", 0.0))
    loss_dens_gamma  = float(getattr(train_cfg, "loss_density_gamma", 0.5))
    n_val_patches    = int(getattr(train_cfg,   "n_val_patches",       8))

    logger.info(f"mesh_lr={mesh_lr}  mesh_hr={mesh_hr}  r={r:.0f}  patch_n={patch_n}")
    logger.info(f"n_steps={n_steps}  patches_per_snap={patches_per_snap}  "
                f"n_per_step={patch_n**3:,}")

    # ── Lagrangian neighbour structure at HR resolution ───────────────────────
    logger.info(f"Precomputing HR Lagrangian neighbours ({mesh_hr}³) …")
    neighbor_idx_hr = get_axis_neighbor_indices(mesh_hr)
    ext_neighbor_idx_hr, ext_offsets_hr, shell_slices_hr = None, None, ()
    if n_shell > 0:
        ext_neighbor_idx_hr, ext_offsets_hr, shell_slices_hr = \
            get_shell_neighbor_indices(mesh_hr, n_shell)
        K = (2 * n_shell + 1)**3 - 1
        logger.info(f"  Extended neighbourhood: n_shell={n_shell}  K={K}  pool='{env_pool_mode}'")
    logger.info("  Done.")

    # ── Model ─────────────────────────────────────────────────────────────────
    lag_model = make_lagrangian_corrector(
        hidden_dim=hidden_dim, n_layers=n_layers, output_dim=3
    )

    # ── Optional CNN prior ────────────────────────────────────────────────────
    cnn_model, cnn_params = None, None
    cnn_ckpt = getattr(model_cfg, "cnn_checkpoint", None)
    if cnn_ckpt:
        logger.info(f"Loading CNN checkpoint: {cnn_ckpt}")
        cnn_model, cnn_params = _load_cnn_massres_checkpoint(str(cnn_ckpt))
        n_cnn = sum(x.size for x in jax.tree_util.tree_leaves(cnn_params))
        logger.info(f"  CNN loaded OK  ({n_cnn:,} params)")

    _apply_cnn_jit = None
    if cnn_model is not None:
        _apply_cnn_jit = jax.jit(
            lambda pos_lr, pos_hr_patch, a: compute_cnn_at_hr_patch(
                cnn_model, cnn_params, pos_lr, pos_hr_patch, mesh_lr, a, use_pm_pot
            )
        )

    # ── Load training snapshot ────────────────────────────────────────────────
    logger.info(f"Loading training snapshot  sim={sim_train}  snap={snap_train}")
    snap_tr = load_subregion_snapshot(
        data_dir, sim_train, snap_train, mesh_lr, mesh_hr, box_size,
        load_velocities=True,
    )
    pos_lr_tr   = snap_tr["pos_lr"]
    pos_hr_tr   = snap_tr["pos_hr"]
    delta_f_tr  = snap_tr["delta_f"]
    vel_hr_tr   = snap_tr.get("vel_hr", jnp.zeros_like(pos_hr_tr))
    a_tr        = snap_tr["a"]

    logger.info(f"  a={a_tr:.4f}  N_lr={pos_lr_tr.shape[0]:,}  N_hr={pos_hr_tr.shape[0]:,}")

    # Compute HR features for the full training snapshot (done once)
    logger.info("Computing HR Lagrangian features …")
    feats_hr_tr, det_D_hr_tr = compute_hr_features(
        pos_hr_tr, neighbor_idx_hr, mesh_hr, mesh_lr,
        use_strain, use_invariants,
        ext_neighbor_idx_hr, ext_offsets_hr, env_pool_mode, shell_slices_hr,
    )
    feat_dim = int(feats_hr_tr.shape[1])
    sc_frac  = float(np.mean(jax.device_get(det_D_hr_tr) < 0))
    logger.info(f"  feat_dim={feat_dim}  SC_hr={sc_frac:.2%}")

    # Compute sample weights for the full snapshot (used to weight patch loss)
    # weights_hr_tr[i] is the loss weight for HR particle i
    pos_hr_mh = pos_hr_tr * r   # mesh_hr units for density estimation
    weights_hr_tr = compute_sample_weights(
        pos_hr_mh, det_D_hr_tr, mesh_hr,
        sc_boost=loss_sc_boost,
        density_boost=loss_dens_boost,
        density_gamma=loss_dens_gamma,
    )

    # CNN correction for full training snapshot (for MLP target adjustment)
    if _apply_cnn_jit is not None:
        logger.info("Computing CNN correction on full training snapshot …")
        cnn_delta_f_tr = _apply_cnn_jit(
            pos_lr_tr, jnp.mod(pos_hr_tr, mesh_lr), jnp.array(a_tr)
        )
        logger.info(f"  CNN |ΔF| mean = "
                    f"{float(jnp.mean(jnp.sqrt(jnp.sum(cnn_delta_f_tr**2, axis=-1)))):.4e}")
    else:
        cnn_delta_f_tr = jnp.zeros_like(delta_f_tr)

    # MLP target: ΔF_mass − ΔF_CNN  (residual for stage-2; equals ΔF_mass for stage-1)
    target_tr = delta_f_tr - cnn_delta_f_tr

    # Pre-convert to numpy for fast patch slicing in the training loop
    feats_hr_np   = np.asarray(jax.device_get(feats_hr_tr))    # [mesh_hr³, D]
    target_np     = np.asarray(jax.device_get(target_tr))       # [mesh_hr³, 3]
    vel_hr_np     = np.asarray(jax.device_get(vel_hr_tr))       # [mesh_hr³, 3]
    weights_np    = np.asarray(jax.device_get(weights_hr_tr))   # [mesh_hr³]
    pos_hr_np     = np.asarray(jax.device_get(pos_hr_tr))       # [mesh_hr³, 3]

    # ── Optimizer & model init ─────────────────────────────────────────────────
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=lr_val,
        warmup_steps=warmup, decay_steps=n_steps,
    )
    optimizer = optax.adamw(schedule, weight_decay=weight_decay)

    rng0        = jax.random.PRNGKey(seed)
    dummy_feats = feats_hr_np[:patch_n**3]
    dummy_vel   = vel_hr_np[:patch_n**3] if use_velocity else np.zeros((patch_n**3, 3))
    params      = lag_model.init(rng0, jnp.array(dummy_feats), jnp.array(dummy_vel), jnp.array(a_tr))
    opt_state   = optimizer.init(params)

    n_mlp = sum(x.size for x in jax.tree_util.tree_leaves(params))
    logger.info(f"MLP: {n_mlp:,} params  |  feat_dim={feat_dim}  hidden={hidden_dim}×{n_layers}")

    train_step_fn = make_train_step(lag_model, optimizer)

    # ── Training loop ─────────────────────────────────────────────────────────
    rng  = jax.random.PRNGKey(seed + 1)
    best_improv = -np.inf
    best_params = None

    for step in range(n_steps):
        rng, patch_rng = jax.random.split(rng)

        # Sample random Lagrangian patch
        patch_idx = sample_lag_patch(patch_rng, mesh_hr, patch_n)   # [patch_n³]

        # Slice all per-particle arrays to the patch
        feats_p   = jnp.array(feats_hr_np[patch_idx])   # [patch_n³, D]
        target_p  = jnp.array(target_np[patch_idx])     # [patch_n³, 3]
        weights_p = jnp.array(weights_np[patch_idx])    # [patch_n³]
        vel_p     = (jnp.array(vel_hr_np[patch_idx])
                     if use_velocity else jnp.zeros((patch_n**3, 3)))

        # Train step
        params, opt_state, loss = train_step_fn(
            params, opt_state, feats_p, vel_p, jnp.array(a_tr), target_p, weights_p
        )

        # ── Logging ───────────────────────────────────────────────────────────
        if step % log_every == 0 or step == n_steps - 1:
            loss_val = float(loss)

            # Pearson R on the training patch
            pred_p_np = np.asarray(jax.device_get(
                lag_model.apply(params, feats_p, vel_p, jnp.array(a_tr))
            ))
            tgt_p_np = np.asarray(jax.device_get(target_p))
            r_vals = [pearsonr(pred_p_np[:, c], tgt_p_np[:, c])[0] for c in range(3)]
            r_mean = float(np.mean(r_vals))

            log = {
                "train/loss":          loss_val,
                "train/pearson_r_mean": r_mean,
                "train/step":          step,
                "train/patch_sc_frac": float(np.mean(
                    jax.device_get(det_D_hr_tr)[patch_idx] < 0
                )),
            }
            wandb.log(log, step=step)
            logger.info(
                f"step {step:5d} | loss={loss_val:.4e} | R={r_mean:.4f}"
            )

        # ── Checkpoint ────────────────────────────────────────────────────────
        if step % save_every == 0 and step > 0:
            with open(out_dir / "checkpoint.pkl", "wb") as fh:
                pickle.dump({
                    "params":    hk.data_structures.to_mutable_dict(params),
                    "model_cfg": vars(model_cfg),
                    "data_cfg":  vars(data_cfg),
                    "step":      step,
                }, fh)

        # ── Validation ────────────────────────────────────────────────────────
        if step % log_every == 0 and step > 0:
            try:
                vlog, improv = val_metrics_subregion(
                    lag_model, params, cnn_model, cnn_params,
                    data_dir, sim_val, snaps_val,
                    mesh_lr, mesh_hr, box_size,
                    neighbor_idx_hr, use_strain, use_invariants, use_velocity, use_pm_pot,
                    n_val_patches=n_val_patches, patch_n=patch_n,
                    ext_neighbor_idx=ext_neighbor_idx_hr,
                    ext_offsets=ext_offsets_hr,
                    env_pool_mode=env_pool_mode,
                    shell_slices=shell_slices_hr,
                )
                wandb.log(vlog, step=step)
                logger.info(f"  val improvement: {improv:.2%}")

                if improv > best_improv:
                    best_improv = improv
                    best_params = hk.data_structures.to_mutable_dict(params)
                    with open(out_dir / "best_params.pkl", "wb") as fh:
                        pickle.dump(best_params, fh)
                    logger.info(f"  ✓ New best  improvement={best_improv:.2%}")
            except Exception as e:
                logger.warning(f"  Validation error: {e}")

    # ── Final save ────────────────────────────────────────────────────────────
    with open(out_dir / "final_params.pkl", "wb") as fh:
        pickle.dump(hk.data_structures.to_mutable_dict(params), fh)
    with open(out_dir / "checkpoint.pkl", "wb") as fh:
        pickle.dump({
            "params":    hk.data_structures.to_mutable_dict(params),
            "model_cfg": vars(model_cfg),
            "data_cfg":  vars(data_cfg),
            "step":      n_steps - 1,
        }, fh)

    logger.info(f"Training done. Output: {out_dir}")
    logger.info(f"Best val improvement: {best_improv:.2%}")
    wandb.finish()


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train sub-region force corrector")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()
    train(args.config)
