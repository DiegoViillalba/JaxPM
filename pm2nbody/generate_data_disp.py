"""
generate_data_disp.py — Data generation for mass-resolution force/displacement correction.

Scheme
------
Generates paired LR / HR N-body simulations where BOTH use the SAME PM mesh:
  - LR: mesh_lr³ particles on mesh_hr³ PM mesh  (e.g. 64³ particles, 128³ mesh)
  - HR: mesh_hr³ particles on mesh_hr³ PM mesh  (e.g. 128³ particles, 128³ mesh)

Using the same PM mesh is critical: it ensures the ONLY difference between LR and
HR is mass resolution (shot noise, small halos), NOT force resolution.  The force
correction learned by train_lag_massres.py captures purely the effect of having
more particles in the density field.

Both simulations share the same large-scale modes (HR linear field downsampled to
LR via box-filtering), ensuring a well-defined Lagrangian particle correspondence:

  LR particle at Lagrangian index (ix, iy, iz)
  ↔ HR particle at Lagrangian index (ix·stride, iy·stride, iz·stride)
  where stride = mesh_hr // n_per_side_lr  (= mesh_hr // mesh_lr)

These two particles start at the SAME physical position.  However, their
trajectories diverge during evolution because the LR density field (from 64³
particles) and the HR density field (from 128³ particles) are different
realizations — the LR particle is NOT a downsampled copy of the HR particle.
It is an independent realization with lower mass resolution.

Particle ordering
-----------------
Both simulations use indexing='ij' in meshgrid (ix outermost, iz innermost).
This matches the convention in train_lag_disp.get_lagrangian_positions() so that
pos[i] and q[i] refer to the same particle.

Output files (all positions in Mpc/h)
--------------------------------------
  pos_m{mesh_lr}_s{n}.npy   [n_snapshots, mesh_lr³, 3]   LR positions
  vel_m{mesh_lr}_s{n}.npy   [n_snapshots, mesh_lr³, 3]   LR velocities
  pos_m{mesh_hr}_s{n}.npy   [n_snapshots, mesh_hr³, 3]   HR positions
  vel_m{mesh_hr}_s{n}.npy   [n_snapshots, mesh_hr³, 3]   HR velocities
  scale_factors.npy          [n_snapshots]                 a values

Usage
-----
  # Quick test (CPU-friendly, ~5 min, ~300 MB):
  python generate_data_disp.py --mode test

  # Production (server, ~2–4 h, ~20 GB):
  python generate_data_disp.py --mode full --out /path/to/data
"""

import argparse
from pathlib import Path
from functools import partial

import scipy
import numpy as np
import jax
import jax.numpy as jnp
import jax_cosmo as jc
from jax import config
from jax.experimental.ode import odeint

from jaxpm.pm import linear_field, lpt, make_ode_fn
from jaxpm.kernels import fftk
from jaxpm.painting import cic_paint

config.update("jax_enable_x64", True)


# ==============================================================================
# Cosmology & power spectrum
# ==============================================================================

def get_linear_field(mesh_shape, box_size, omega_c, sigma8, seed=0):
    """Sample a Gaussian random field with Planck15 power spectrum."""
    k  = jnp.logspace(-4, 1, 128)
    pk = jc.power.linear_matter_power(jc.Planck15(Omega_c=omega_c, sigma8=sigma8), k)

    def pk_fn(x):
        xflat = x.reshape(-1)
        idx   = jnp.clip(jnp.searchsorted(k, xflat), 1, k.size - 1)
        k0, k1 = k[idx - 1], k[idx]
        p0, p1 = pk[idx - 1], pk[idx]
        t = (xflat - k0) / (k1 - k0 + 1e-12)
        return (p0 + t * (p1 - p0)).reshape(x.shape)

    return linear_field(mesh_shape, box_size, pk_fn, seed=jax.random.PRNGKey(seed))


def downsample_field(field, downsampling_factor=2):
    """Box-filter downsampling — same large-scale modes, reduced resolution."""
    f  = (downsampling_factor,) * 3
    w  = np.ones(f) / np.prod(f)
    sm = scipy.ndimage.convolve(field, w, mode="mirror")
    return sm[::downsampling_factor, ::downsampling_factor, ::downsampling_factor]


# ==============================================================================
# Particle initialisation  (indexing='ij' — matches train_lag_disp convention)
# ==============================================================================

def make_lagrangian_grid(n_per_side, mesh_shape):
    """
    Regular Lagrangian grid with indexing='ij' (ix outermost, iz innermost).

    Returns [n_per_side³, 3] in mesh units.  Particle flat index j maps to
    Lagrangian index (ix=j//N², iy=(j//N)%N, iz=j%N).
    """
    coords = jnp.arange(n_per_side, dtype=jnp.float64)
    ix, iy, iz = jnp.meshgrid(
        coords * mesh_shape[0] / n_per_side,
        coords * mesh_shape[1] / n_per_side,
        coords * mesh_shape[2] / n_per_side,
        indexing="ij",   # ← critical: matches train_lag_disp.get_lagrangian_positions
    )
    return jnp.stack([ix.ravel(), iy.ravel(), iz.ravel()], axis=-1)


def get_ics(n_per_side, mesh_shape, lin_field, snapshot, omega_c, sigma8):
    """2LPT initial conditions. Returns (pos, vel) in mesh units."""
    particles = make_lagrangian_grid(n_per_side, mesh_shape)
    cosmo     = jc.Planck15(Omega_c=omega_c, sigma8=sigma8)
    dx, p, _  = lpt(cosmo, lin_field, particles, snapshot)
    return particles + dx, p


# ==============================================================================
# Simulation runner
# ==============================================================================

@partial(jax.jit, static_argnums=(0,))
def run_simulation(n_mesh, omega_c, sigma8, initial_conditions, snapshots):
    cosmo = jc.Planck15(Omega_c=omega_c, sigma8=sigma8)
    return odeint(
        make_ode_fn((n_mesh, n_mesh, n_mesh)),
        initial_conditions,
        snapshots,
        cosmo,
        rtol=1e-5,
        atol=1e-5,
    )


# ==============================================================================
# Main
# ==============================================================================

PRESETS = {
    # ── Quick local test ───────────────────────────────────────────────────
    # LR: 64³ = 262k particles   HR: 128³ = 2M particles
    # ~5-10 min on CPU, ~300 MB disk
    "test": dict(
        mesh_lr     = 64,
        mesh_hr     = 128,
        box_size    = 128.0,   # Mpc/h
        n_snapshots = 10,
        n_sims      = 2,
        out_dir     = "./data_disp_test",
    ),
    # ── Medium — server with ~16 GB VRAM ─────────────────────────────────
    # LR: 128³ = 2M particles   HR: 256³ = 16M particles
    # ~1-2 h, ~5 GB disk
    "medium": dict(
        mesh_lr     = 128,
        mesh_hr     = 256,
        box_size    = 256.0,
        n_snapshots = 20,
        n_sims      = 4,
        out_dir     = "./data_disp_medium",
    ),
    # ── Full production ───────────────────────────────────────────────────
    "full": dict(
        mesh_lr     = 128,
        mesh_hr     = 256,
        box_size    = 256.0,
        n_snapshots = 50,
        n_sims      = 10,
        out_dir     = "./data_disp_full",
    ),
}


def generate(
    mesh_lr:     int,
    mesh_hr:     int,
    box_size:    float,
    n_snapshots: int,
    n_sims:      int,
    out_dir:     str,
    omega_c:     float = 0.25,
    sigma8:      float = 0.8,
):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    stride        = mesh_hr // mesh_lr
    n_per_side_lr = mesh_lr
    n_per_side_hr = mesh_hr
    mesh_shape_lr = (mesh_lr,) * 3
    mesh_shape_hr = (mesh_hr,) * 3
    box           = [box_size] * 3
    snapshots     = jnp.linspace(0.1, 1.0, n_snapshots)

    print(f"{'='*55}")
    print(f"  Mass-resolution data generation")
    print(f"  PM mesh (both sims): {mesh_hr}³  ← same force resolution")
    print(f"  LR: {mesh_lr}³ = {mesh_lr**3:,} particles  on  {mesh_hr}³ PM mesh")
    print(f"  HR: {mesh_hr}³ = {mesh_hr**3:,} particles  on  {mesh_hr}³ PM mesh")
    print(f"  stride = {stride}  |  box = {box_size} Mpc/h  |  {n_snapshots} snaps  |  {n_sims} sims")
    print(f"  Output → {out_path.resolve()}")
    print(f"{'='*55}\n")

    # Save scale factors once
    np.save(out_path / "scale_factors.npy", np.asarray(snapshots))

    for n in range(n_sims):
        print(f"── Sim {n} / {n_sims - 1} {'─'*35}")

        # ── Linear fields: HR first, then downsample to LR ────────────────
        print("  [1/4] Linear field HR …")
        lin_hr = get_linear_field(mesh_shape_hr, box, omega_c, sigma8, seed=n)
        lin_lr = downsample_field(np.asarray(lin_hr), downsampling_factor=stride)
        lin_lr = jnp.array(lin_lr)

        # ── ICs ───────────────────────────────────────────────────────────
        print(f"  [2/4] ICs  LR ({n_per_side_lr}³) …")
        ics_lr = get_ics(n_per_side_lr, mesh_shape_lr, lin_lr, snapshots[0], omega_c, sigma8)

        print(f"  [2/4] ICs  HR ({n_per_side_hr}³) …")
        ics_hr = get_ics(n_per_side_hr, mesh_shape_hr, lin_hr, snapshots[0], omega_c, sigma8)

        del lin_hr, lin_lr   # free before running sims

        # ── Simulations ───────────────────────────────────────────────────
        # CRITICAL: both use mesh_hr as PM mesh so that force resolution is
        # identical — the only difference is the number of particles (mass res).
        print(f"  [3/4] ODE  LR ({n_per_side_lr}³ particles on {mesh_hr}³ PM mesh) …")
        pos_lr, vel_lr = run_simulation(mesh_hr, omega_c, sigma8, ics_lr, snapshots)

        print(f"  [3/4] ODE  HR ({n_per_side_hr}³ particles on {mesh_hr}³ PM mesh) …")
        pos_hr, vel_hr = run_simulation(mesh_hr, omega_c, sigma8, ics_hr, snapshots)

        del ics_lr, ics_hr

        # ── Save (convert mesh units → Mpc/h) ────────────────────────────
        # Both simulations ran on mesh_hr PM mesh, so positions are in
        # [0, mesh_hr) mesh units.  Divide by mesh_hr for both to get Mpc/h.
        # load_snapshot loads with scale = mesh_lr/box, giving positions in
        # [0, mesh_lr) range (= mesh_hr units / r), consistent with r = mesh_hr/mesh_lr.
        print("  [4/4] Saving …")
        np.save(out_path / f"pos_m{mesh_lr}_s{n}.npy",
                np.asarray(pos_lr) / mesh_hr * box_size)   # ← divide by mesh_hr (PM mesh)
        np.save(out_path / f"vel_m{mesh_lr}_s{n}.npy",
                np.asarray(vel_lr) / mesh_hr * box_size)
        np.save(out_path / f"pos_m{mesh_hr}_s{n}.npy",
                np.asarray(pos_hr) / mesh_hr * box_size)
        np.save(out_path / f"vel_m{mesh_hr}_s{n}.npy",
                np.asarray(vel_hr) / mesh_hr * box_size)

        # Verify Lagrangian ordering: initial displacement should be small
        # Initial q in LR mesh units = (ix, iy, iz) for ix in range(mesh_lr)
        # Positions saved as /mesh_hr*box, loaded as *mesh_lr/box = /r range
        q_lr_0 = np.stack(np.meshgrid(
            *[np.arange(n_per_side_lr) * box_size / n_per_side_lr] * 3,
            indexing="ij",
        ), axis=-1).reshape(-1, 3)   # initial q in Mpc/h
        pos_lr_mpc = np.asarray(pos_lr)[0] / mesh_hr * box_size   # snap-0 in Mpc/h
        disp_check = np.mean(np.abs(pos_lr_mpc - q_lr_0))
        print(f"  ✓ saved  |  mean |Ψ_LR| at snap 0: {disp_check:.4f} Mpc/h")

        del pos_lr, vel_lr, pos_hr, vel_hr

        # File sizes
        for fname in [f"pos_m{mesh_lr}_s{n}.npy", f"pos_m{mesh_hr}_s{n}.npy"]:
            size_mb = (out_path / fname).stat().st_size / 1e6
            print(f"       {fname}  {size_mb:.1f} MB")

    print(f"\nDone. All files in {out_path.resolve()}")
    print("Config for lag_disp_mlp.yaml:")
    print(f"  data_dir:    {out_path.resolve()}")
    print(f"  mesh_lr:     {mesh_lr}")
    print(f"  mesh_hr:     {mesh_hr}")
    print(f"  box_size:    {box_size}")
    print(f"  n_particles: {mesh_lr}   # per dimension")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate LR/HR displacement SR data")
    parser.add_argument(
        "--mode", choices=list(PRESETS), default="test",
        help="Preset: test (64/128, fast), medium (128/256, server), full (128/256, production)",
    )
    parser.add_argument("--out",     default=None,  help="Override output directory")
    parser.add_argument("--mesh_lr", default=None,  type=int)
    parser.add_argument("--mesh_hr", default=None,  type=int)
    parser.add_argument("--L",       default=None,  type=float, help="Box size Mpc/h")
    parser.add_argument("--snaps",   default=None,  type=int,   help="Number of snapshots")
    parser.add_argument("--n_sims",  default=None,  type=int,   help="Number of simulations")
    args = parser.parse_args()

    params = dict(PRESETS[args.mode])
    if args.out     is not None: params["out_dir"]     = args.out
    if args.mesh_lr is not None: params["mesh_lr"]     = args.mesh_lr
    if args.mesh_hr is not None: params["mesh_hr"]     = args.mesh_hr
    if args.L       is not None: params["box_size"]    = args.L
    if args.snaps   is not None: params["n_snapshots"] = args.snaps
    if args.n_sims  is not None: params["n_sims"]      = args.n_sims

    generate(**params)
