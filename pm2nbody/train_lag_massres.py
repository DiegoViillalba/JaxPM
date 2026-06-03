"""
train_lag_massres.py — Lagrangian MLP for mass-resolution force correction.

Conceptual difference vs. force-resolution pipeline (train_lag_force.py)
-------------------------------------------------------------------------
  Force-resolution:   same N_lr particles, different mesh (128³ vs 256³ mesh)
                      → corrects missing small-scale FORCE MODES
  Mass-resolution:    different N particles (128³ vs 256³), each on its own mesh
                      → corrects missing MASS elements + their forces

Target
------
  ΔF_mass_i = F_HR(q_i) − F_LR_i

  F_LR_i    = PM force on LR particle i  from  n_part³  density on  mesh_lr  grid
  F_HR(q_i) = PM force on HR particle at the SAME Lagrangian position q_i
              obtained by:
                1. Run 256³ HR simulation (done by generate_data_disp.py)
                2. Sort HR particles to Lagrangian C-order
                3. Take every stride-th particle: [::stride, ::stride, ::stride]
              This particle shares the same initial position but evolved under
              a density field sampled with (stride)³× more particles.

Physical effects captured (beyond force-resolution correction)
--------------------------------------------------------------
  • Reduced shot noise in the density field
  • Small halos resolved only at HR mass resolution
  • Accurate halo profiles and mass functions
  • Correct small-scale tidal fields from resolved sub-halos

Data: output of generate_data_disp.py  (same format)
  pos_m{mesh_lr}_s{n}.npy   [n_snaps, n_part³,  3]   LR positions  (Mpc/h)
  pos_m{mesh_hr}_s{n}.npy   [n_snaps, mesh_hr³, 3]   HR positions  (Mpc/h)
  scale_factors.npy

Usage
-----
  python train_lag_massres.py --config configs/lag_massres.yaml

Two-stage workflow
------------------
  Stage 1 (optional): train_cnn_massres.py   →  CNN-WST baseline
  Stage 2:            train_lag_massres.py   →  MLP residual
    set model.cnn_checkpoint in lag_massres.yaml to activate stage 2
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

from jaxpm.kernels import fftk
from jaxpm.pm import get_delta, potential_kgrid_to_force_at_pos
from jaxpm.lagrangian import (
    get_axis_neighbor_indices,
    get_shell_neighbor_indices,
    compute_deformation_tensor,
    make_lagrangian_corrector,
)

# Re-use helpers from the existing pipelines
from train_lag_force import (
    load_snapshot,
    snapshot_features,
    make_train_step,
    compute_sample_weights,
    compute_metrics,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")


# ==============================================================================
# 1. Mass-resolution force pair
# ==============================================================================

def _gaussian_filter_k(delta_k: jnp.ndarray, kvec: list, sigma_cells: float) -> jnp.ndarray:
    """
    Apply an isotropic Gaussian low-pass filter in k-space.

    W(k) = exp(-½ |k|² σ²)

    Parameters
    ----------
    delta_k     : [..., N//2+1]   density field in k-space (from rfftn)
    kvec        : list of k-vector arrays from fftk()
                  each has shape broadcastable to delta_k
    sigma_cells : Gaussian σ in real-space mesh-cell units
                  σ=0 → no filtering (identity)
                  σ=1 → suppresses Nyquist mode by exp(-½π²) ≈ 0.007

    Notes
    -----
    k from fftk() is in rad/cell:  k_max ≈ π  (Nyquist)
    Gaussian: W(k) = exp(-½ k² σ²_cells)
    At Nyquist (k=π), σ=1:  W = exp(-π²/2) ≈ 0.007  (strongly attenuated)
    At Nyquist (k=π), σ=0.5: W = exp(-π²/8) ≈ 0.29   (moderate)
    """
    k2 = kvec[0] ** 2 + kvec[1] ** 2 + kvec[2] ** 2   # (N,N,N//2+1) via broadcast
    return delta_k * jnp.exp(-0.5 * k2 * sigma_cells ** 2)


@partial(jax.jit, static_argnums=(2, 3, 4, 5, 6))
def compute_massres_force_pair(
    pos_lr: jnp.ndarray,
    pos_hr: jnp.ndarray,
    n_part: int,
    mesh_lr: int,
    mesh_hr: int,
    smooth_sigma_lr: float = 0.0,
    smooth_sigma_hr: float = 0.0,
) -> tuple:
    """
    Compute (F_LR, F_HR_at_q, ΔF_mass) in mesh_lr force units.

    Physical setup
    --------------
    Both LR (n_part³) and HR (mesh_hr³) simulations run on the SAME PM mesh of
    size mesh_hr.  The ONLY difference between them is mass resolution:
      • LR: n_part³ particles  (e.g. 64³)  on mesh_hr³ PM mesh
      • HR: mesh_hr³ particles (e.g. 128³) on mesh_hr³ PM mesh

    This means:
      - Both share the same force resolution (same k_max, same kernel)
      - They differ only in shot noise and small-halo contribution to the density
      - ΔF captures the statistical force difference due to mass resolution,
        NOT force-resolution effects (which are identical for both)

    Lagrangian correspondence
    -------------------------
    Both IC grids are built with make_lagrangian_grid using indexing='ij'.
    LR particle grid spacing = mesh_hr/n_part = stride  (in mesh_hr units).
    HR particle grid spacing = 1  (in mesh_hr units).

    LR particle m = ix·n_part² + iy·n_part + iz  starts at:
        q_LR = (ix·stride, iy·stride, iz·stride)  in mesh_hr units

    HR particle M = IX·mesh_hr² + IY·mesh_hr + IZ  starts at:
        q_HR = (IX, IY, IZ)  in mesh_hr units

    For the HR particle at (IX=ix·stride, IY=iy·stride, IZ=iz·stride):
        q_HR = q_LR  ✓  — same initial physical position

    The stride-subsampling picks exactly these matching HR particles:
        f_hr_all.reshape(mesh_hr, mesh_hr, mesh_hr, 3)[::stride, ::stride, ::stride, :]
    so f_hr_at_q[m] and f_lr[m] correspond to the same Lagrangian origin. ✓

    IMPORTANT: although q_LR[m] = q_HR[stride·m] initially, the trajectories diverge
    because each realization evolves under a different density field (64³ vs 128³
    particles).  The LR particle IS NOT a subsampled copy of the HR particle — it
    is an independent realization with the same large-scale modes but lower mass
    resolution. ΔF therefore captures the STATISTICAL force difference between
    realizations, not a deterministic per-particle difference.

    Force unit conversion
    ---------------------
    Both LR and HR forces are computed on mesh_hr, so they come out in mesh_hr
    gradient units.  We convert to mesh_lr force units via:
        F_mesh_lr = F_mesh_hr × (mesh_hr / mesh_lr) = F_mesh_hr × r
    This keeps the output scale consistent with the Lagrangian MLP's feature space
    (which is built in mesh_lr units).

    Gaussian shot-noise smoothing  (both sigmas in LR-cell units)
    -----------------------------
    smooth_sigma_lr : smooth LR density field before computing F_LR.
                      In mesh_hr cells: sigma_lr_hr = smooth_sigma_lr × r.
                      Shot noise scale for LR ≈ stride = r mesh_hr cells.
                      Smoothing at σ ≈ r removes it but also removes real signal.
                      Usually leave at 0 (LR force is "as simulated" at inference).
    smooth_sigma_hr : smooth HR density field before computing F_HR.
                      In mesh_hr cells: sigma_hr_hr = smooth_sigma_hr × r.
                      Shot noise scale for HR ≈ 1 mesh_hr cell.
                      Recommended: smooth_sigma_hr = 0.5 → σ = r×0.5 mesh_hr cells
                      (= 1 HR cell for r=2) removes Nyquist noise from HR target.

    Parameters
    ----------
    pos_lr          : [n_part³, 3]   LR positions in mesh_lr units
    pos_hr          : [mesh_hr³, 3]  HR positions in mesh_lr units
                      load_snapshot converts both to mesh_lr units:
                        pos * (mesh_lr / box_size)
                      so pos * r gives mesh_hr units for both.
    n_part          : int    LR particles per dimension  (mesh_hr must be divisible)
    mesh_lr, mesh_hr: int    mesh_lr < mesh_hr;  stride = mesh_hr // n_part
    smooth_sigma_lr : float  Gaussian σ for LR field in LR-cell units (default 0)
    smooth_sigma_hr : float  Gaussian σ for HR field in LR-cell units (default 0)

    Returns
    -------
    f_lr      : [n_part³, 3]   LR PM force  (mesh_lr force units)
    f_hr_at_q : [n_part³, 3]   HR PM force at same Lagrangian positions as LR
    delta_f   : [n_part³, 3]   = f_hr_at_q − f_lr
    """
    r      = float(mesh_hr) / float(mesh_lr)
    stride = mesh_hr // n_part

    # Shared PM mesh k-vectors (mesh_hr for both — same PM resolution)
    kvec_pm = fftk((mesh_hr,) * 3)

    # ── LR force: n_part³ particles painted on the shared mesh_hr PM mesh ────
    # Both pos_lr and pos_hr are in mesh_lr units from load_snapshot.
    # Multiply by r to convert to mesh_hr units, then paint on mesh_hr grid.
    pos_lr_pm  = jnp.mod(pos_lr * r, mesh_hr)   # [n_part³, 3] in mesh_hr units
    delta_lr_k = jnp.fft.rfftn(get_delta(pos_lr_pm, (mesh_hr,) * 3))
    if smooth_sigma_lr > 0.0:
        # sigma in LR-cell units → mesh_hr-cell units: × r
        delta_lr_k = _gaussian_filter_k(delta_lr_k, kvec_pm, smooth_sigma_lr * r)
    f_lr_pm = potential_kgrid_to_force_at_pos(delta_lr_k, pos_lr_pm, kvec_pm)
    # Convert mesh_hr force units → mesh_lr force units
    f_lr = f_lr_pm * r   # [n_part³, 3]

    # ── HR force: mesh_hr³ particles painted on the shared mesh_hr PM mesh ───
    pos_hr_pm  = jnp.mod(pos_hr * r, mesh_hr)   # [mesh_hr³, 3] in mesh_hr units
    delta_hr_k = jnp.fft.rfftn(get_delta(pos_hr_pm, (mesh_hr,) * 3))
    if smooth_sigma_hr > 0.0:
        # sigma in LR-cell units → mesh_hr-cell units: × r
        delta_hr_k = _gaussian_filter_k(delta_hr_k, kvec_pm, smooth_sigma_hr * r)
    f_hr_all = potential_kgrid_to_force_at_pos(delta_hr_k, pos_hr_pm, kvec_pm)
    # Convert mesh_hr force units → mesh_lr force units
    f_hr_all_lr = f_hr_all * r   # [mesh_hr³, 3]

    # ── Stride-subsample HR forces at LR Lagrangian positions ─────────────────
    # HR particles maintain Lagrangian C-order (ix outermost) from data generation.
    # Flat index M = IX·mesh_hr² + IY·mesh_hr + IZ → reshape to (mesh_hr,mesh_hr,mesh_hr,3)
    # [::stride, ::stride, ::stride] picks HR particle at q = (ix·stride, iy·stride, iz·stride)
    # which matches LR particle m = ix·n_part² + iy·n_part + iz.  ✓
    f_hr_at_q = f_hr_all_lr.reshape(mesh_hr, mesh_hr, mesh_hr, 3)[
        ::stride, ::stride, ::stride, :
    ].reshape(-1, 3)   # [n_part³, 3], same ordering as pos_lr

    return f_lr, f_hr_at_q, f_hr_at_q - f_lr


@partial(jax.jit, static_argnums=(2, 3, 4, 5))
def compute_subregion_force_pair(
    pos_lr: jnp.ndarray,
    pos_hr: jnp.ndarray,
    mesh_lr: int,
    mesh_hr: int,
    smooth_sigma_hr: float = 0.0,
    smooth_sigma_lr: float = 0.0,
) -> tuple:
    """
    Compute the force pair for the sub-region pipeline.

    Unlike compute_massres_force_pair (which stride-subsamples HR forces to match LR
    Lagrangian positions), this function evaluates forces at ALL HR particle positions.
    This allows patch-based training on any spatial sub-region.

    Physical setup
    --------------
    Same as compute_massres_force_pair: both LR and HR simulations run on the shared
    mesh_hr PM mesh, so force resolution is identical.

    Force evaluation
    ----------------
    • f_lr_at_hr : force from LR density (on mesh_hr), evaluated at ALL HR positions
    • f_hr        : force from HR density (on mesh_hr), evaluated at all HR positions
    • delta_f     : f_hr − f_lr_at_hr  (mass-resolution correction at each HR position)

    Returned shapes: [mesh_hr³, 3], in mesh_lr force units.

    Parameters
    ----------
    pos_lr          : [mesh_lr³, 3]  LR positions in mesh_lr units
    pos_hr          : [mesh_hr³, 3]  HR positions in mesh_lr units (from load_snapshot)
    mesh_lr, mesh_hr: int            mesh sizes; mesh_hr > mesh_lr
    smooth_sigma_hr : float          Gaussian σ for HR density (in LR-cell units, default 0)
    smooth_sigma_lr : float          Gaussian σ for LR density (in LR-cell units, default 0)
    """
    r      = float(mesh_hr) / float(mesh_lr)
    kvec   = fftk((mesh_hr,) * 3)

    # ── LR density on mesh_hr → force at HR positions ─────────────────────
    pos_lr_pm  = jnp.mod(pos_lr * r, float(mesh_hr))
    delta_lr_k = jnp.fft.rfftn(get_delta(pos_lr_pm, (mesh_hr,) * 3))
    if smooth_sigma_lr > 0.0:
        delta_lr_k = _gaussian_filter_k(delta_lr_k, kvec, smooth_sigma_lr * r)

    pos_hr_pm  = jnp.mod(pos_hr * r, float(mesh_hr))   # HR positions in mesh_hr units
    f_lr_at_hr = potential_kgrid_to_force_at_pos(delta_lr_k, pos_hr_pm, kvec) * r

    # ── HR density on mesh_hr → force at HR positions ─────────────────────
    delta_hr_k = jnp.fft.rfftn(get_delta(pos_hr_pm, (mesh_hr,) * 3))
    if smooth_sigma_hr > 0.0:
        delta_hr_k = _gaussian_filter_k(delta_hr_k, kvec, smooth_sigma_hr * r)
    f_hr = potential_kgrid_to_force_at_pos(delta_hr_k, pos_hr_pm, kvec) * r

    return f_lr_at_hr, f_hr, f_hr - f_lr_at_hr


# ==============================================================================
# 2. Optional CNN mass-resolution correction  (stage-1 prior)
# ==============================================================================

def compute_cnn_massres_correction(
    cnn_model,
    cnn_params,
    pos_lr_t: jnp.ndarray,
    vel_lr_t: jnp.ndarray,
    a: float,
    mesh_lr: int,
) -> jnp.ndarray:
    """
    CNN-WST mass-resolution force correction  ΔF_CNN = +∇ΔΦ_CNN(x_LR).
    Identical architecture to the force-resolution CNN.
    Returns [N, 3] in mesh_lr units.
    """
    pos_lr_mod = jnp.mod(pos_lr_t, mesh_lr)
    delta      = get_delta(pos_lr_mod, (mesh_lr,) * 3)
    delta_k    = jnp.fft.rfftn(delta)
    kvec       = fftk((mesh_lr,) * 3)
    _, pm_pot  = potential_kgrid_to_force_at_pos(
        delta_k, pos_lr_mod, kvec, return_potential=True
    )
    grid_data = jnp.stack([pm_pot, delta], axis=-1)
    vel_sg    = jax.lax.stop_gradient(vel_lr_t)

    def phi_sum(pos):
        return jnp.sum(cnn_model.apply(cnn_params, grid_data, pos, a, vel_sg)[:, 0])

    return jax.grad(phi_sum)(pos_lr_t)


# ==============================================================================
# 3. CNN checkpoint loader
# ==============================================================================

def _load_cnn_massres_checkpoint(ckpt_path: str):
    """
    Load a CNN checkpoint saved by train_cnn_massres.py.

    The checkpoint format is:
        {"params": ..., "model_cfg": {...}, "data_cfg": {...}}

    Accepts either:
      - a directory path  → looks for <dir>/checkpoint.pkl
      - a direct .pkl file path

    Returns (model, immutable_params).
    """
    from train_cnn_massres import build_cnn_model  # local import to avoid circular

    p = Path(ckpt_path)
    if p.is_dir():
        p = p / "checkpoint.pkl"
    if not p.exists():
        raise FileNotFoundError(f"CNN checkpoint not found: {p}")

    with open(p, "rb") as fh:
        ckpt = pickle.load(fh)

    if "model_cfg" not in ckpt:
        raise KeyError(
            f"Checkpoint at {p} does not contain 'model_cfg'. "
            "Make sure it was saved by train_cnn_massres.py."
        )

    model_cfg_ns = SimpleNamespace(**ckpt["model_cfg"])
    model        = build_cnn_model(model_cfg_ns)
    params       = hk.data_structures.to_immutable_dict(ckpt["params"])
    return model, params


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
        project = wb_cfg.get("project", "pm2nbody_massres"),
        name    = getattr(exp_cfg, "name", None),
        tags    = wb_cfg.get("tags", []),
        config  = cfg,
    )
    out_dir = Path(getattr(exp_cfg, "output_dir", "runs/lag_massres")) / wandb.run.name
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

    use_strain     = bool(getattr(model_cfg, "use_strain",     True))
    use_invariants = bool(getattr(model_cfg, "use_invariants", False))
    use_velocity   = bool(getattr(model_cfg, "use_velocity",   True))
    n_shell        = int(getattr(model_cfg,  "n_shell",        0))
    env_pool_mode  = str(getattr(model_cfg,  "env_pool_mode",  "mean_var"))
    hidden_dim     = int(getattr(model_cfg,  "hidden_dim",     64))
    n_layers       = int(getattr(model_cfg,  "n_layers",       3))

    n_steps      = int(getattr(train_cfg,  "n_steps",       1000))
    lr_val       = float(getattr(train_cfg, "lr",            3e-4))
    weight_decay = float(getattr(train_cfg, "weight_decay",  1e-4))
    warmup       = int(getattr(train_cfg,   "warmup_steps",   50))
    log_every    = int(getattr(train_cfg,   "log_every",      50))
    save_every   = int(getattr(train_cfg,   "save_every",    200))
    seed         = int(getattr(train_cfg,   "seed",            0))

    loss_sc_boost      = float(getattr(train_cfg, "loss_sc_boost",      1.0))
    loss_density_boost = float(getattr(train_cfg, "loss_density_boost", 0.0))
    loss_density_gamma = float(getattr(train_cfg, "loss_density_gamma", 0.5))

    # Gaussian smoothing for force pair (in LR-cell units)
    smooth_sigma_lr = float(getattr(data_cfg, "smooth_sigma_lr", 0.0))
    smooth_sigma_hr = float(getattr(data_cfg, "smooth_sigma_hr", 0.0))
    if smooth_sigma_lr > 0 or smooth_sigma_hr > 0:
        logger.info(
            f"Force smoothing: σ_lr={smooth_sigma_lr} LR-cells  "
            f"σ_hr={smooth_sigma_hr} LR-cells (={smooth_sigma_hr * mesh_hr / mesh_lr:.2f} HR-cells)"
        )

    # ── Lagrangian neighbour structure ────────────────────────────────────────
    logger.info(f"Precomputing Lagrangian neighbours  {n_part}³")
    neighbor_idx = get_axis_neighbor_indices(n_part)
    ext_neighbor_idx, ext_offsets, shell_slices = None, None, ()
    if n_shell > 0:
        ext_neighbor_idx, ext_offsets, shell_slices = get_shell_neighbor_indices(
            n_part, n_shell
        )
        logger.info(f"Extended neighbourhood: n_shell={n_shell}  pool='{env_pool_mode}'")

    def get_feats(pos_t):
        return snapshot_features(
            pos_t, neighbor_idx, mesh_lr, use_strain, use_invariants,
            ext_neighbor_idx, ext_offsets, env_pool_mode, shell_slices,
        )

    # ── Model ─────────────────────────────────────────────────────────────────
    lag_model = make_lagrangian_corrector(
        hidden_dim=hidden_dim, n_layers=n_layers, output_dim=3
    )

    # ── Optional CNN prior (stage 1) ──────────────────────────────────────────
    cnn_model, cnn_params = None, None
    cnn_ckpt = getattr(model_cfg, "cnn_checkpoint", None)
    if cnn_ckpt:
        logger.info(f"Loading CNN checkpoint: {cnn_ckpt}")
        try:
            cnn_model, cnn_params = _load_cnn_massres_checkpoint(cnn_ckpt)
            logger.info(f"  CNN loaded OK  ({sum(x.size for x in jax.tree_util.tree_leaves(cnn_params)):,} params)")
        except Exception as e:
            logger.error(f"Could not load CNN: {e}")
            raise

    _apply_cnn = None
    if cnn_model is not None:
        _apply_cnn = jax.jit(
            lambda pos, vel, a: compute_cnn_massres_correction(
                cnn_model, cnn_params, pos, vel, a, mesh_lr
            )
        )

    # ── Force pair helper (captures sigma from config) ───────────────────────
    _force_pair = lambda pl, ph: compute_massres_force_pair(
        pl, ph, n_part, mesh_lr, mesh_hr, smooth_sigma_lr, smooth_sigma_hr
    )

    # ── Training snapshot ─────────────────────────────────────────────────────
    logger.info(f"Loading  sim={sim_train}  snap={snap_train}")
    pos_lr_t, vel_lr_t, pos_hr_t, a_train = load_snapshot(
        data_dir, sim_train, snap_train, mesh_lr, mesh_hr, box_size
    )
    if pos_hr_t is None:
        raise RuntimeError("HR positions required.  Check data_dir / mesh_hr in config.")
    logger.info(f"  a={a_train:.4f}  N_lr={pos_lr_t.shape[0]:,}  N_hr={pos_hr_t.shape[0]:,}")

    # Features
    feats_tr, det_D_tr = get_feats(pos_lr_t)
    feat_dim = int(feats_tr.shape[1])
    sc_frac  = float(np.mean(det_D_tr < 0))
    logger.info(f"  feat_dim={feat_dim}  SC={sc_frac:.2%}")

    # Force pair
    logger.info("Computing mass-resolution force pair …")
    f_lr_tr, f_hr_tr, delta_f_tr = _force_pair(pos_lr_t, pos_hr_t)
    df_mag = float(jnp.mean(jnp.sqrt(jnp.sum(delta_f_tr ** 2, axis=-1))))
    f_mag  = float(jnp.mean(jnp.sqrt(jnp.sum(f_hr_tr   ** 2, axis=-1))))
    logger.info(f"  |F_HR| mean = {f_mag:.4e}   |ΔF_mass| mean = {df_mag:.4e}  "
                f"ratio = {df_mag / (f_mag + 1e-12):.3f}")

    target_tr = delta_f_tr
    if _apply_cnn is not None:
        logger.info("Subtracting CNN correction from target …")
        f_cnn = _apply_cnn(pos_lr_t, vel_lr_t, jnp.array(a_train))
        target_tr = delta_f_tr - f_cnn
        logger.info(f"  |ΔF_CNN|={float(jnp.mean(jnp.sqrt(jnp.sum(f_cnn**2,axis=-1)))):.4e}  "
                    f"|residual|={float(jnp.mean(jnp.sqrt(jnp.sum(target_tr**2,axis=-1)))):.4e}")

    vel_feat_tr = vel_lr_t if use_velocity else jnp.zeros_like(vel_lr_t)

    # Per-particle weights
    use_weighted = loss_sc_boost > 1.0 or loss_density_boost > 0.0
    if use_weighted:
        weights_np = compute_sample_weights(
            pos_lr_t, det_D_tr, mesh_lr,
            sc_boost=loss_sc_boost,
            density_boost=loss_density_boost,
            density_gamma=loss_density_gamma,
        )
        logger.info(f"  weights: min={weights_np.min():.3f}  max={weights_np.max():.3f}")
    else:
        weights_np = np.ones(pos_lr_t.shape[0], dtype=np.float32)
    weights_tr = jnp.array(weights_np)

    # ── Init model ────────────────────────────────────────────────────────────
    rng    = jax.random.PRNGKey(seed)
    params = lag_model.init(rng, feats_tr, vel_feat_tr, jnp.array(a_train))
    n_p    = sum(x.size for x in jax.tree_util.tree_leaves(params))
    logger.info(f"Parameters: {n_p:,}")

    wandb.log({
        "model/n_params": n_p, "model/feat_dim": feat_dim,
        "data/df_mag": df_mag, "data/f_hr_mag": f_mag,
        "data/sc_frac": sc_frac,
        "data/smooth_sigma_lr": smooth_sigma_lr,
        "data/smooth_sigma_hr": smooth_sigma_hr,
    }, step=0)

    # ── Optimiser ─────────────────────────────────────────────────────────────
    lr_sched  = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=lr_val,
        warmup_steps=warmup, decay_steps=n_steps, end_value=lr_val * 0.01,
    )
    optimizer  = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(lr_sched, weight_decay=weight_decay),
    )
    opt_state  = optimizer.init(params)
    train_step = make_train_step(lag_model, optimizer)

    # ── Loop ──────────────────────────────────────────────────────────────────
    best_val, best_params = float("inf"), None

    for step in range(1, n_steps + 1):
        params, opt_state, loss = train_step(
            params, opt_state, feats_tr, vel_feat_tr, jnp.array(a_train),
            target_tr, weights_tr,
        )

        if step % log_every == 0 or step == 1:
            pred_tr  = jax.jit(lag_model.apply)(
                params, feats_tr, vel_feat_tr, jnp.array(a_train)
            )
            log_dict = compute_metrics(pred_tr, target_tr, det_D_tr, "train/")
            log_dict["train/loss"] = float(loss)
            log_dict["train/lr"]   = float(lr_sched(step))
            del pred_tr

            # Validation
            val_mses = []
            for vsnap in snaps_val:
                vpos, vvel, vpos_hr, va = load_snapshot(
                    data_dir, sim_val, vsnap, mesh_lr, mesh_hr, box_size
                )
                vfeats, vdet_D = get_feats(vpos)
                _, _, vdf      = _force_pair(vpos, vpos_hr)
                vtarget = vdf
                if _apply_cnn is not None:
                    vtarget = vdf - _apply_cnn(vpos, vvel, jnp.array(va))
                vvel_feat = vvel if use_velocity else jnp.zeros_like(vvel)
                vpred     = jax.jit(lag_model.apply)(
                    params, vfeats, vvel_feat, jnp.array(va)
                )
                vmet = compute_metrics(vpred, vtarget, vdet_D, f"val/snap{vsnap}/")
                val_mses.append(vmet[f"val/snap{vsnap}/force_mse"])
                log_dict.update(vmet)
                del vpos, vvel, vfeats, vtarget, vpred

            mean_val = float(np.mean(val_mses))
            log_dict["val/force_mse_mean"] = mean_val
            wandb.log(log_dict, step=step)
            logger.info(
                f"step {step:5d}  loss={float(loss):.4e}"
                f"  R̄={log_dict['train/pearson_r_mean']:.3f}"
                f"  val_mse={mean_val:.4e}"
            )

            if mean_val < best_val:
                best_val   = mean_val
                best_params = jax.device_get(hk.data_structures.to_mutable_dict(params))

        if step % save_every == 0:
            ckpt = out_dir / f"params_step{step:05d}.pkl"
            with open(ckpt, "wb") as fh:
                pickle.dump(jax.device_get(hk.data_structures.to_mutable_dict(params)), fh)
            logger.info(f"  checkpoint → {ckpt}")

    # ── Save ──────────────────────────────────────────────────────────────────
    if best_params is not None:
        with open(out_dir / "best_params.pkl", "wb") as fh:
            pickle.dump(best_params, fh)
    with open(out_dir / "final_params.pkl", "wb") as fh:
        pickle.dump(jax.device_get(hk.data_structures.to_mutable_dict(params)), fh)

    wandb.log({"val/best_force_mse": best_val}, step=n_steps)
    logger.info(f"Done. best_val_mse={best_val:.4e}  → {out_dir}")
    wandb.finish()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    train(p.parse_args().config)
