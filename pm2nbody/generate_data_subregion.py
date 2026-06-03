"""
generate_data_subregion.py — Data generation for sub-region force prediction.

Generates the same LR/HR N-body simulation pairs as generate_data_disp.py, and
additionally pre-computes and saves force fields at ALL HR particle positions.
These pre-computed files enable efficient patch-based training in
train_subregion_force.py without repeating the expensive FFT force computation
at every training step.

Extra output files (in mesh_lr force units):
  delta_f_m{mesh_hr}_s{n}.npy   [n_snaps, mesh_hr³, 3]  ΔF = F_HR − F_LR@HR
  f_hr_m{mesh_hr}_s{n}.npy      [n_snaps, mesh_hr³, 3]  F_HR at all HR positions
  f_lr_at_hr_m{mesh_hr}_s{n}.npy [n_snaps, mesh_hr³, 3] F_LR at all HR positions

Position / velocity files (same format as generate_data_disp.py, Mpc/h):
  pos_m{mesh_lr}_s{n}.npy  [n_snaps, mesh_lr³, 3]
  vel_m{mesh_lr}_s{n}.npy  [n_snaps, mesh_lr³, 3]
  pos_m{mesh_hr}_s{n}.npy  [n_snaps, mesh_hr³, 3]
  vel_m{mesh_hr}_s{n}.npy  [n_snaps, mesh_hr³, 3]
  scale_factors.npy         [n_snaps]

Physical setup (identical to generate_data_disp.py):
  LR: mesh_lr³ particles on mesh_hr³ PM mesh  (e.g. 64³ particles, 128³ mesh)
  HR: mesh_hr³ particles on mesh_hr³ PM mesh  (e.g. 128³ particles, 128³ mesh)
  Both share the same large-scale IC modes (HR field box-filtered to LR).

Force convention:
  All forces are in mesh_lr units, consistent with load_snapshot (train_lag_force.py)
  and compute_subregion_force_pair (train_lag_massres.py).

Usage:
  # Quick test on CPU (~5–10 min, ~400 MB with forces):
  python generate_data_subregion.py --mode test

  # Production:
  python generate_data_subregion.py --mode full --out /path/to/data
"""

import argparse
import sys
from pathlib import Path
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import config

config.update("jax_enable_x64", True)

# ── Reuse simulation generation from generate_data_disp ───────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_data_disp import (
    get_linear_field,
    downsample_field,
    make_lagrangian_grid,
    get_ics,
    run_simulation,
)

# ── Force pair for sub-region (no stride subsampling) ─────────────────────────
from train_lag_massres import compute_subregion_force_pair


# ==============================================================================
# Presets
# ==============================================================================

PRESETS = {
    # ── Quick local test ───────────────────────────────────────────────────
    # ~5-10 min on CPU, ~400 MB disk (includes force fields)
    "test": dict(
        mesh_lr     = 64,
        mesh_hr     = 128,
        box_size    = 128.0,
        n_snapshots = 10,
        n_sims      = 2,
        out_dir     = "./data_subregion_test",
    ),
    # ── Medium — server with ~16 GB VRAM ──────────────────────────────────
    "medium": dict(
        mesh_lr     = 128,
        mesh_hr     = 256,
        box_size    = 256.0,
        n_snapshots = 20,
        n_sims      = 4,
        out_dir     = "./data_subregion_medium",
    ),
    # ── Full production ───────────────────────────────────────────────────
    "full": dict(
        mesh_lr     = 128,
        mesh_hr     = 256,
        box_size    = 256.0,
        n_snapshots = 50,
        n_sims      = 10,
        out_dir     = "./data_subregion_full",
    ),
}


# ==============================================================================
# Force field computation
# ==============================================================================

def _compute_forces_for_snapshot(
    pos_lr_sim: np.ndarray,
    pos_hr_sim: np.ndarray,
    mesh_lr: int,
    mesh_hr: int,
    smooth_sigma_hr: float,
) -> tuple:
    """
    Compute sub-region force pair for one snapshot.

    Parameters
    ----------
    pos_lr_sim : [mesh_lr³, 3]  LR positions in mesh_hr simulation units [0, mesh_hr)
    pos_hr_sim : [mesh_hr³, 3]  HR positions in mesh_hr simulation units [0, mesh_hr)
    mesh_lr, mesh_hr            mesh sizes
    smooth_sigma_hr             Gaussian σ for HR density (in LR-cell units)

    Returns
    -------
    f_lr_at_hr : [mesh_hr³, 3]  LR force at HR positions, mesh_lr units
    f_hr       : [mesh_hr³, 3]  HR force at HR positions, mesh_lr units
    delta_f    : [mesh_hr³, 3]  f_hr − f_lr_at_hr, mesh_lr units
    """
    r = float(mesh_hr) / float(mesh_lr)
    # Positions saved in simulation are in [0, mesh_hr) (mesh_hr units).
    # compute_subregion_force_pair expects positions in mesh_lr units.
    pos_lr_ml = jnp.array(pos_lr_sim / r, dtype=jnp.float64)
    pos_hr_ml = jnp.array(pos_hr_sim / r, dtype=jnp.float64)

    f_lr_at_hr, f_hr, delta_f = compute_subregion_force_pair(
        pos_lr_ml, pos_hr_ml,
        mesh_lr, mesh_hr,
        smooth_sigma_hr, 0.0,   # smooth HR, keep LR as-simulated
    )
    return (
        np.asarray(jax.device_get(f_lr_at_hr), dtype=np.float32),
        np.asarray(jax.device_get(f_hr),       dtype=np.float32),
        np.asarray(jax.device_get(delta_f),    dtype=np.float32),
    )


# ==============================================================================
# Main generation function
# ==============================================================================

def generate(
    mesh_lr:        int,
    mesh_hr:        int,
    box_size:       float,
    n_snapshots:    int,
    n_sims:         int,
    out_dir:        str,
    omega_c:        float = 0.25,
    sigma8:         float = 0.8,
    smooth_sigma_hr: float = 0.5,
    save_components: bool = True,   # also save f_hr and f_lr_at_hr separately
):
    """
    Generate LR/HR simulation pairs and pre-compute force fields.

    Parameters
    ----------
    smooth_sigma_hr : Gaussian σ for HR force target (LR-cell units).
                      Recommended 0.5 → 1 HR-cell smoothing for r=2.
                      Must match the value used in configs/subregion_force.yaml.
    save_components : if True, also save f_hr and f_lr_at_hr in addition to delta_f.
    """
    out_path  = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    stride        = mesh_hr // mesh_lr
    n_per_side_lr = mesh_lr
    n_per_side_hr = mesh_hr
    mesh_shape_lr = (mesh_lr,) * 3
    mesh_shape_hr = (mesh_hr,) * 3
    box           = [box_size] * 3
    snapshots     = jnp.linspace(0.1, 1.0, n_snapshots)

    print(f"{'='*60}")
    print(f"  Sub-region force prediction — data generation")
    print(f"  PM mesh (both sims): {mesh_hr}³  ← same force resolution")
    print(f"  LR: {mesh_lr}³ = {mesh_lr**3:,} particles  on  {mesh_hr}³ PM mesh")
    print(f"  HR: {mesh_hr}³ = {mesh_hr**3:,} particles  on  {mesh_hr}³ PM mesh")
    print(f"  stride={stride}  |  box={box_size} Mpc/h  |  {n_snapshots} snaps  |  {n_sims} sims")
    print(f"  σ_hr={smooth_sigma_hr} LR-cells (={smooth_sigma_hr * stride:.1f} HR-cells)")
    print(f"  Output → {out_path.resolve()}")
    print(f"{'='*60}\n")

    np.save(out_path / "scale_factors.npy", np.asarray(snapshots))

    # Pre-compile force computation for this mesh pair (done on first call)
    print("JIT-compiling force computation …")
    _force_jit = jax.jit(
        partial(compute_subregion_force_pair,
                mesh_lr=mesh_lr, mesh_hr=mesh_hr,
                smooth_sigma_hr=smooth_sigma_hr, smooth_sigma_lr=0.0)
    )

    for n in range(n_sims):
        print(f"\n── Sim {n} / {n_sims - 1} {'─'*40}")

        # ── Linear fields ─────────────────────────────────────────────────
        print("  [1/5] Linear field HR …")
        lin_hr = get_linear_field(mesh_shape_hr, box, omega_c, sigma8, seed=n)
        lin_lr = downsample_field(np.asarray(lin_hr), downsampling_factor=stride)
        lin_lr = jnp.array(lin_lr)

        # ── ICs ───────────────────────────────────────────────────────────
        print(f"  [2/5] ICs LR ({n_per_side_lr}³) …")
        ics_lr = get_ics(n_per_side_lr, mesh_shape_lr, lin_lr, snapshots[0], omega_c, sigma8)
        print(f"  [2/5] ICs HR ({n_per_side_hr}³) …")
        ics_hr = get_ics(n_per_side_hr, mesh_shape_hr, lin_hr, snapshots[0], omega_c, sigma8)
        del lin_hr, lin_lr

        # ── Simulations (both on mesh_hr PM mesh) ─────────────────────────
        print(f"  [3/5] ODE LR ({n_per_side_lr}³ particles, {mesh_hr}³ PM mesh) …")
        pos_lr_all, vel_lr_all = run_simulation(mesh_hr, omega_c, sigma8, ics_lr, snapshots)
        print(f"  [3/5] ODE HR ({n_per_side_hr}³ particles, {mesh_hr}³ PM mesh) …")
        pos_hr_all, vel_hr_all = run_simulation(mesh_hr, omega_c, sigma8, ics_hr, snapshots)
        del ics_lr, ics_hr

        # ── Save positions / velocities (same format as generate_data_disp) ──
        print("  [4/5] Saving positions …")
        pos_lr_np = np.asarray(pos_lr_all)   # [n_snaps, mesh_lr³, 3]  mesh_hr sim units
        vel_lr_np = np.asarray(vel_lr_all)
        pos_hr_np = np.asarray(pos_hr_all)   # [n_snaps, mesh_hr³, 3]
        vel_hr_np = np.asarray(vel_hr_all)

        # Divide by mesh_hr to get Mpc/h (same convention as generate_data_disp)
        np.save(out_path / f"pos_m{mesh_lr}_s{n}.npy",
                (pos_lr_np / mesh_hr * box_size).astype(np.float32))
        np.save(out_path / f"vel_m{mesh_lr}_s{n}.npy",
                (vel_lr_np / mesh_hr * box_size).astype(np.float32))
        np.save(out_path / f"pos_m{mesh_hr}_s{n}.npy",
                (pos_hr_np / mesh_hr * box_size).astype(np.float32))
        np.save(out_path / f"vel_m{mesh_hr}_s{n}.npy",
                (vel_hr_np / mesh_hr * box_size).astype(np.float32))

        # ── Pre-compute force fields per snapshot ──────────────────────────
        # pos_lr_np is in [0, mesh_hr) mesh_hr units.
        # compute_subregion_force_pair expects mesh_lr units → divide by r.
        print(f"  [5/5] Pre-computing force fields ({n_snapshots} snaps) …")
        r = float(mesh_hr) / float(mesh_lr)

        df_arr          = np.zeros((n_snapshots, mesh_hr**3, 3), dtype=np.float32)
        f_hr_arr        = np.zeros_like(df_arr)
        f_lr_at_hr_arr  = np.zeros_like(df_arr)

        for s in range(n_snapshots):
            pos_lr_s = jnp.array(pos_lr_np[s] / r, dtype=jnp.float64)  # mesh_lr units
            pos_hr_s = jnp.array(pos_hr_np[s] / r, dtype=jnp.float64)  # mesh_lr units

            f_lr_at_hr_s, f_hr_s, delta_f_s = _force_jit(pos_lr_s, pos_hr_s)

            f_lr_at_hr_arr[s] = np.asarray(jax.device_get(f_lr_at_hr_s), dtype=np.float32)
            f_hr_arr[s]       = np.asarray(jax.device_get(f_hr_s),       dtype=np.float32)
            df_arr[s]         = np.asarray(jax.device_get(delta_f_s),    dtype=np.float32)

            if (s + 1) % 5 == 0 or s == n_snapshots - 1:
                print(f"       snap {s+1}/{n_snapshots}  "
                      f"|ΔF| mean = {np.sqrt(np.sum(df_arr[s]**2, axis=-1)).mean():.3e}")

        np.save(out_path / f"delta_f_m{mesh_hr}_s{n}.npy",       df_arr)
        if save_components:
            np.save(out_path / f"f_hr_m{mesh_hr}_s{n}.npy",      f_hr_arr)
            np.save(out_path / f"f_lr_at_hr_m{mesh_hr}_s{n}.npy", f_lr_at_hr_arr)

        del pos_lr_all, vel_lr_all, pos_hr_all, vel_hr_all
        del pos_lr_np, vel_lr_np, pos_hr_np, vel_hr_np
        del df_arr, f_hr_arr, f_lr_at_hr_arr

        # File sizes
        for fname in [f"pos_m{mesh_lr}_s{n}.npy", f"pos_m{mesh_hr}_s{n}.npy",
                      f"delta_f_m{mesh_hr}_s{n}.npy"]:
            fpath = out_path / fname
            if fpath.exists():
                print(f"       {fname}  {fpath.stat().st_size / 1e6:.1f} MB")

    print(f"\nDone. All files in {out_path.resolve()}")
    print(f"\nConfig snippet for subregion_force.yaml:")
    print(f"  data_dir:         {out_path.resolve()}")
    print(f"  mesh_lr:          {mesh_lr}")
    print(f"  mesh_hr:          {mesh_hr}")
    print(f"  box_size:         {box_size}")
    print(f"  n_particles:      {mesh_lr}   # LR particles per dimension")
    print(f"  smooth_sigma_hr:  {smooth_sigma_hr}")


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate LR/HR sim pairs + pre-computed HR force fields"
    )
    parser.add_argument("--mode", choices=list(PRESETS), default="test",
                        help="Preset: test (64/128, fast), medium (128/256, server), full")
    parser.add_argument("--out",     default=None,   help="Override output directory")
    parser.add_argument("--mesh_lr", default=None,   type=int)
    parser.add_argument("--mesh_hr", default=None,   type=int)
    parser.add_argument("--L",       default=None,   type=float, help="Box size Mpc/h")
    parser.add_argument("--snaps",   default=None,   type=int,   help="Number of snapshots")
    parser.add_argument("--n_sims",  default=None,   type=int)
    parser.add_argument("--sigma_hr", default=0.5,   type=float,
                        help="Gaussian σ_hr for HR force (LR-cell units, default 0.5)")
    parser.add_argument("--no_components", action="store_true",
                        help="Skip saving f_hr and f_lr_at_hr (only save delta_f)")
    args = parser.parse_args()

    params = dict(PRESETS[args.mode])
    if args.out     is not None: params["out_dir"]     = args.out
    if args.mesh_lr is not None: params["mesh_lr"]     = args.mesh_lr
    if args.mesh_hr is not None: params["mesh_hr"]     = args.mesh_hr
    if args.L       is not None: params["box_size"]    = args.L
    if args.snaps   is not None: params["n_snapshots"] = args.snaps
    if args.n_sims  is not None: params["n_sims"]      = args.n_sims
    params["smooth_sigma_hr"] = args.sigma_hr
    params["save_components"] = not args.no_components

    generate(**params)
