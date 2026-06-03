"""
generate_data_single.py — Single N-body simulation data generator.

Runs one N-body simulation with N³ particles on an N³ PM mesh.
Saves positions and velocities — no LR/HR comparison, no pre-computed forces.
Forces are computed on-the-fly during training (train_subregion_forceres.py).

Output files (positions in Mpc/h):
  pos_m{n_part}_s{sim_id}.npy  [n_snapshots, n_part³, 3]
  vel_m{n_part}_s{sim_id}.npy  [n_snapshots, n_part³, 3]
  scale_factors.npy            [n_snapshots]

Usage:
  python generate_data_single.py --mode test
  python generate_data_single.py --n_part 128 --out ./my_data
"""

import argparse
from pathlib import Path
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import jax_cosmo as jc
from jax import config
from jax.experimental.ode import odeint

from jaxpm.pm import linear_field, lpt, make_ode_fn
from jaxpm.kernels import fftk

config.update("jax_enable_x64", True)


# ==============================================================================
# Helpers (reused from generate_data_disp.py)
# ==============================================================================

def get_linear_field(mesh_shape, box_size, omega_c, sigma8, seed=0):
    import jax_cosmo as jc
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


@partial(jax.jit, static_argnums=(0,))
def run_simulation(n_mesh, omega_c, sigma8, ics, snapshots):
    cosmo = jc.Planck15(Omega_c=omega_c, sigma8=sigma8)
    return odeint(make_ode_fn((n_mesh,) * 3), ics, snapshots, cosmo,
                  rtol=1e-5, atol=1e-5)


PRESETS = {
    "test": dict(n_part=64,  box_size=128.0, n_snapshots=10, n_sims=2,
                 out_dir="./data_single_test"),
    "medium": dict(n_part=128, box_size=256.0, n_snapshots=20, n_sims=4,
                   out_dir="./data_single_medium"),
    "full":   dict(n_part=256, box_size=512.0, n_snapshots=50, n_sims=10,
                   out_dir="./data_single_full"),
}


def generate(n_part, box_size, n_snapshots, n_sims, out_dir,
             omega_c=0.25, sigma8=0.8):
    out_path  = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    mesh_shape = (n_part,) * 3
    box        = [box_size] * 3
    snapshots  = jnp.linspace(0.1, 1.0, n_snapshots)

    print(f"{'='*55}")
    print(f"  Single-simulation data generation")
    print(f"  {n_part}³ = {n_part**3:,} particles  on  {n_part}³ PM mesh")
    print(f"  box={box_size} Mpc/h  |  {n_snapshots} snaps  |  {n_sims} sims")
    print(f"  Output → {out_path.resolve()}")
    print(f"{'='*55}\n")

    np.save(out_path / "scale_factors.npy", np.asarray(snapshots))

    for n in range(n_sims):
        print(f"── Sim {n} / {n_sims - 1} {'─'*35}")

        # ── Linear field → 2LPT ICs ────────────────────────────────────────
        print("  [1/3] Linear field + ICs …")
        lin = get_linear_field(mesh_shape, box, omega_c, sigma8, seed=n)

        coords  = jnp.arange(n_part, dtype=jnp.float64)
        ix, iy, iz = jnp.meshgrid(coords, coords, coords, indexing="ij")
        particles = jnp.stack([ix.ravel(), iy.ravel(), iz.ravel()], axis=-1)
        cosmo     = jc.Planck15(Omega_c=omega_c, sigma8=sigma8)
        dx, p, _  = lpt(cosmo, lin, particles, snapshots[0])
        del lin
        ics = particles + dx, p

        # ── ODE ───────────────────────────────────────────────────────────
        print(f"  [2/3] ODE  {n_part}³ particles …")
        pos_all, vel_all = run_simulation(n_part, omega_c, sigma8, ics, snapshots)

        # ── Save (Mpc/h) ───────────────────────────────────────────────────
        print("  [3/3] Saving …")
        np.save(out_path / f"pos_m{n_part}_s{n}.npy",
                (np.asarray(pos_all) / n_part * box_size).astype(np.float32))
        np.save(out_path / f"vel_m{n_part}_s{n}.npy",
                (np.asarray(vel_all) / n_part * box_size).astype(np.float32))

        del pos_all, vel_all
        size_mb = (out_path / f"pos_m{n_part}_s{n}.npy").stat().st_size / 1e6
        print(f"       pos_m{n_part}_s{n}.npy  {size_mb:.1f} MB")

    print(f"\nDone. Files in {out_path.resolve()}")
    print(f"\nConfig snippet for subregion_forceres.yaml:")
    print(f"  data_dir:    {out_path.resolve()}")
    print(f"  n_part:      {n_part}")
    print(f"  mesh_lr:     {n_part // 2}   # coarse force mesh (example: n_part / 2)")
    print(f"  mesh_hr:     {n_part}        # fine force mesh  (= n_part)")
    print(f"  box_size:    {box_size}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",   choices=list(PRESETS), default="test")
    parser.add_argument("--out",    default=None)
    parser.add_argument("--n_part", default=None, type=int)
    parser.add_argument("--L",      default=None, type=float)
    parser.add_argument("--snaps",  default=None, type=int)
    parser.add_argument("--n_sims", default=None, type=int)
    args = parser.parse_args()

    params = dict(PRESETS[args.mode])
    if args.out    is not None: params["out_dir"]     = args.out
    if args.n_part is not None: params["n_part"]      = args.n_part
    if args.L      is not None: params["box_size"]    = args.L
    if args.snaps  is not None: params["n_snapshots"] = args.snaps
    if args.n_sims is not None: params["n_sims"]      = args.n_sims
    generate(**params)
