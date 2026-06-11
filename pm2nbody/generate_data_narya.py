"""
generate_data_narya.py — JaxPM simulation with Narya (L256N768) cosmology.

Generates N-body simulation data for force-resolution correction training using
the same cosmological parameters as the BaccoSims Narya run (L=256 Mpc/h, N=768).
Output data is compatible with train_pretrain_forceres.py and train_subregion_forceres.py.

Cosmology (from LG3_param.txt-usedvalues):
  Omega0=0.36, OmegaLambda=0.64, OmegaBaryon=0, h=0.7
  As=1.84444e-9, ns=1.01, w0=-1, wa=0
  sigma8 ≈ 0.78 (derived from As via CAMB at Narya parameters)

Memory strategy:
  Batch ODE (default for N<=128): store all snapshots at once, fast.
  Sequential ODE (default for N>=256): integrate snapshot-by-snapshot,
    keeping only the current state on GPU. Allows large N on single GPU.

Output files (positions in Mpc/h, n_part units in brackets):
  pos_m{n_part}_s{sim_id}.npy  [n_snapshots, n_part³, 3]
  vel_m{n_part}_s{sim_id}.npy  [n_snapshots, n_part³, 3]
  scale_factors.npy            [n_snapshots]
  cosmo_params.json            cosmology used (for reproducibility)

Recommended meshes:
  N=128 → mesh_lr=64,  mesh_hr=128  (quick validation)
  N=256 → mesh_lr=128, mesh_hr=256  (production, ~1 Mpc/h force res)
  N=384 → mesh_lr=192, mesh_hr=384  (ambitious, ~0.67 Mpc/h)

Usage:
  python generate_data_narya.py --mode narya256 --n_sims 5
  python generate_data_narya.py --n_part 256 --L 256 --n_sims 5 --out ./data_narya
  python generate_data_narya.py --mode narya128  # quick test
"""

import argparse
import json
from pathlib import Path
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import jax_cosmo as jc
from jax.experimental.ode import odeint

from jaxpm.pm import linear_field, lpt, make_ode_fn

import os
os.environ["JAX_ENABLE_X64"] = "0"


# ==============================================================================
# Narya cosmology (BaccoSims L256N768)
# ==============================================================================

NARYA_COSMO = dict(
    Omega_c=0.31,   # = Omega_m(0.36) − Omega_b(0.05); Narya runs pure CDM but
    Omega_b=0.05,   # jax_cosmo Eisenstein-Hu transfer fn divides by Omega_b → needs >0
    h=0.70,
    n_s=1.01,
    sigma8=0.78,   # derived from As=1.84444e-9 at Narya params
    w0=-1.0,
    wa=0.0,
)


# ==============================================================================
# Presets
# ==============================================================================

PRESETS = {
    "narya128": dict(
        n_part=128, box_size=256.0, n_snapshots=10, n_sims=2,
        out_dir="./data_narya128",
        seq_ode=False,   # 128³ fits in batch mode
    ),
    "narya256": dict(
        n_part=256, box_size=256.0, n_snapshots=10, n_sims=5,
        out_dir="./data_narya256",
        seq_ode=True,    # 256³ × 10 snaps needs sequential mode
    ),
    "narya384": dict(
        n_part=384, box_size=256.0, n_snapshots=5, n_sims=3,
        out_dir="./data_narya384",
        seq_ode=True,
    ),
}


# ==============================================================================
# Power spectrum / cosmology helpers
# ==============================================================================

def make_cosmo(**kwargs):
    params = {**NARYA_COSMO, **kwargs}
    return jc.Planck15(
        Omega_c=params["Omega_c"],
        Omega_b=params["Omega_b"],
        h=params["h"],
        n_s=params["n_s"],
        sigma8=params["sigma8"],
        w0=params["w0"],
        wa=params["wa"],
    )


def get_linear_field(mesh_shape, box_size, cosmo_params, seed=0):
    cosmo = make_cosmo(**cosmo_params)
    k  = jnp.logspace(-4, 1, 256)
    pk = jc.power.linear_matter_power(cosmo, k)

    def pk_fn(x):
        xflat = x.reshape(-1)
        idx   = jnp.clip(jnp.searchsorted(k, xflat), 1, k.size - 1)
        k0, k1 = k[idx - 1], k[idx]
        p0, p1 = pk[idx - 1], pk[idx]
        t = (xflat - k0) / (k1 - k0 + 1e-12)
        return (p0 + t * (p1 - p0)).reshape(x.shape)

    return linear_field(mesh_shape, box_size, pk_fn, seed=jax.random.PRNGKey(seed))


# ==============================================================================
# Simulation helpers
# ==============================================================================

@partial(jax.jit, static_argnums=(0, 1))
def run_segment(n_mesh, cosmo_frozen, ics, a_range):
    """Integrate ODE from a_range[0] to a_range[-1], return state at each step.
    cosmo_frozen is a sorted tuple of (key, value) pairs for JIT hashability.
    """
    cosmo = make_cosmo(**dict(cosmo_frozen))
    return odeint(make_ode_fn((n_mesh,) * 3), ics, a_range, cosmo,
                  rtol=1e-5, atol=1e-5)


def run_batch(n_part, cosmo_params, ics, snapshots):
    """Run all snapshots at once (fast, for small N)."""
    cosmo_frozen = tuple(sorted(cosmo_params.items()))
    pos_all, vel_all = run_segment(n_part, cosmo_frozen, ics, snapshots)
    return np.asarray(pos_all), np.asarray(vel_all)


def run_sequential(n_part, cosmo_params, ics, snapshots):
    """Run snapshot-by-snapshot (memory-safe, for large N).

    Integrates from a_{i-1} → a_i for each snapshot, keeping only the
    current state on GPU. GPU peak memory ≈ 2 × n_part³ × 3 × 8 bytes.
    """
    cosmo_frozen = tuple(sorted(cosmo_params.items()))
    pos_list, vel_list = [], []

    pos_cur, vel_cur = ics
    a_prev = snapshots[0]

    for i, a_snap in enumerate(snapshots):
        a_range = jnp.array([a_prev, a_snap]) if i > 0 else jnp.array([a_snap, a_snap])
        if i == 0:
            # At first snapshot: just use ICs (already at a_snap via 2LPT)
            pos_out, vel_out = pos_cur, vel_cur
        else:
            segs = run_segment(n_part, cosmo_frozen, (pos_cur, vel_cur),
                                jnp.array([a_prev, a_snap]))
            pos_out = segs[0][-1]
            vel_out = segs[1][-1]

        pos_list.append(np.asarray(jax.device_get(pos_out)))
        vel_list.append(np.asarray(jax.device_get(vel_out)))
        pos_cur, vel_cur = pos_out, vel_out
        a_prev = a_snap

        if (i + 1) % 5 == 0 or i == len(snapshots) - 1:
            print(f"       snap {i+1}/{len(snapshots)}  a={a_snap:.3f}")

    return np.stack(pos_list, axis=0), np.stack(vel_list, axis=0)


# ==============================================================================
# Main generator
# ==============================================================================

def generate(
    n_part=256,
    box_size=256.0,
    n_snapshots=10,
    n_sims=5,
    out_dir="./data_narya256",
    seq_ode=True,
    a_start=0.1,
    a_end=1.0,
    cosmo_params=None,
):
    if cosmo_params is None:
        cosmo_params = dict(NARYA_COSMO)

    out_path   = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    mesh_shape = (n_part,) * 3
    box        = [box_size] * 3
    snapshots  = np.linspace(a_start, a_end, n_snapshots)
    cosmo_obj  = make_cosmo(**cosmo_params)

    n_particles = n_part ** 3
    # Peak GPU memory estimate per sim (positions only, float32)
    mem_per_snap_mb = n_particles * 3 * 4 / 1e6
    mem_batch_gb    = mem_per_snap_mb * n_snapshots / 1e3

    print(f"{'='*60}")
    print(f"  Narya-cosmology data generation")
    print(f"  {n_part}³ = {n_particles:,} particles  |  L={box_size} Mpc/h")
    print(f"  {n_snapshots} snapshots  a=[{a_start:.2f},{a_end:.2f}]  |  {n_sims} sims")
    omega_m = cosmo_params['Omega_c'] + cosmo_params['Omega_b']
    print(f"  Cosmology: Ω_m={omega_m:.3f} (Ω_c={cosmo_params['Omega_c']}+Ω_b={cosmo_params['Omega_b']})"
          f", h={cosmo_params['h']}, σ₈={cosmo_params['sigma8']}, n_s={cosmo_params['n_s']}")
    print(f"  ODE mode: {'sequential (memory-safe)' if seq_ode else 'batch (fast)'}")
    print(f"  Peak GPU mem/sim: {mem_per_snap_mb:.0f} MB/snap × "
          f"{'1 (seq)' if seq_ode else str(n_snapshots) + ' (batch)'} "
          f"= {mem_per_snap_mb if seq_ode else mem_batch_gb*1e3:.0f} "
          f"{'MB' if seq_ode else 'MB'}")
    print(f"  Output → {out_path.resolve()}")
    print(f"{'='*60}\n")

    np.save(out_path / "scale_factors.npy", snapshots)
    with open(out_path / "cosmo_params.json", "w") as f:
        json.dump({**cosmo_params, "box_size": box_size, "n_part": n_part,
                   "n_snapshots": n_snapshots}, f, indent=2)

    snapshots_jnp = jnp.array(snapshots)

    for n in range(n_sims):
        print(f"── Sim {n} / {n_sims - 1} {'─'*40}")

        print("  [1/3] Linear field + 2LPT ICs …")
        lin = get_linear_field(mesh_shape, box, cosmo_params, seed=n)

        coords    = jnp.arange(n_part, dtype=jnp.float32)
        ix, iy, iz = jnp.meshgrid(coords, coords, coords, indexing="ij")
        particles = jnp.stack([ix.ravel(), iy.ravel(), iz.ravel()], axis=-1)
        dx, p, _  = lpt(cosmo_obj, lin, particles, snapshots[0])
        del lin
        ics = (particles + dx, p)

        print(f"  [2/3] ODE  {n_part}³ particles …")
        if seq_ode:
            pos_all, vel_all = run_sequential(n_part, cosmo_params, ics, snapshots)
        else:
            pos_all, vel_all = run_batch(n_part, cosmo_params, ics, snapshots_jnp)

        print("  [3/3] Saving …")
        pos_mpc = (pos_all / n_part * box_size).astype(np.float32)
        vel_mpc = (vel_all / n_part * box_size).astype(np.float32)

        np.save(out_path / f"pos_m{n_part}_s{n}.npy", pos_mpc)
        np.save(out_path / f"vel_m{n_part}_s{n}.npy", vel_mpc)

        del pos_all, vel_all, pos_mpc, vel_mpc
        size_mb = (out_path / f"pos_m{n_part}_s{n}.npy").stat().st_size / 1e6
        print(f"       pos_m{n_part}_s{n}.npy  {size_mb:.1f} MB")

    print(f"\nDone. Files in {out_path.resolve()}")
    print(f"\nConfig snippet for narya_pretrain.yaml:")
    print(f"  data_dir:    {out_path.resolve()}")
    print(f"  n_part:      {n_part}")
    print(f"  mesh_lr:     {n_part}      # = N: Nyquist-matched PM")
    print(f"  mesh_hr:     {n_part * 2}  # = 2N: 2x oversampling for halo forces")
    print(f"  box_size:    {box_size}")
    print(f"  sim_ids_train: {list(range(n_sims - 1))}")
    print(f"  sim_ids_val:   [{n_sims - 1}]")


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Narya-cosmology simulation data for force correction training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode",    choices=list(PRESETS), default="narya256",
                        help="Preset configuration (default: narya256)")
    parser.add_argument("--out",     default=None,
                        help="Output directory (overrides preset)")
    parser.add_argument("--n_part",  default=None, type=int,
                        help="Particles per dimension (overrides preset)")
    parser.add_argument("--L",       default=None, type=float,
                        help="Box size in Mpc/h (overrides preset)")
    parser.add_argument("--n_sims",  default=None, type=int,
                        help="Number of simulations (overrides preset)")
    parser.add_argument("--snaps",   default=None, type=int,
                        help="Number of snapshots (overrides preset)")
    parser.add_argument("--a_start", default=0.1,  type=float)
    parser.add_argument("--a_end",   default=1.0,  type=float)
    parser.add_argument("--sigma8",  default=None, type=float,
                        help="Override sigma8 (default: 0.78)")
    parser.add_argument("--seq",     action="store_true", default=None,
                        help="Force sequential ODE mode")
    parser.add_argument("--batch",   action="store_true", default=None,
                        help="Force batch ODE mode")
    args = parser.parse_args()

    params = dict(PRESETS[args.mode])
    if args.out    is not None: params["out_dir"]     = args.out
    if args.n_part is not None: params["n_part"]      = args.n_part
    if args.L      is not None: params["box_size"]    = args.L
    if args.n_sims is not None: params["n_sims"]      = args.n_sims
    if args.snaps  is not None: params["n_snapshots"] = args.snaps
    if args.seq:                params["seq_ode"]     = True
    if args.batch:              params["seq_ode"]     = False

    cosmo_params = dict(NARYA_COSMO)
    if args.sigma8 is not None:
        cosmo_params["sigma8"] = args.sigma8

    generate(**params, a_start=args.a_start, a_end=args.a_end,
             cosmo_params=cosmo_params)
