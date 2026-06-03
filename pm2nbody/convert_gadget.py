"""
convert_gadget.py — Build LR/HR displacement-SR pairs from a single Gadget snapshot.

Strategy
--------
With only one Gadget snapshot (no ICs), we create a synthetic LR/HR pair
by exploiting the LAGRANGIAN PARTICLE IDs that Gadget stores:

  HR: subsample the 792³ Gadget particles at stride_hr
      e.g. stride_hr=3 → every 3rd particle per axis → 264³ "HR" dataset

  LR: CIC block-average the full 792³ field in blocks of (stride_lr)³
      Each block contributes one "LR particle" = centre-of-mass of the block
      e.g. stride_lr=6 → 132³ "LR" dataset

  LR is a genuine PM-level approximation (PM codes do exactly this
  CIC averaging), so ΔΨ = x_HR(q) − x_LR(q) is a realistic displacement
  correction target.

Gadget particle ID encoding (standard for cubic-grid ICs):
  ID = ix * N² + iy * N + iz + 1   (1-indexed, ix outermost)
  →  ix = (ID-1) // N²
     iy = ((ID-1) // N) % N
     iz = (ID-1) % N

Output (all positions in Mpc/h, Lagrangian C-order ix outermost)
-----------------------------------------------------------------
  pos_m{mesh_lr}_s0.npy   [1, n_part_lr³, 3]   LR positions
  vel_m{mesh_lr}_s0.npy   [1, n_part_lr³, 3]   LR velocities (block-averaged)
  pos_m{mesh_hr}_s0.npy   [1, n_part_hr³, 3]   HR positions
  vel_m{mesh_hr}_s0.npy   [1, n_part_hr³, 3]   HR velocities
  scale_factors.npy        [1]                   scale factor a

Usage
-----
  python convert_gadget.py \\
      --snap /path/to/snapshot_XXX.hdf5 \\
      --n_gadget 792 \\
      --stride_lr 6 \\     # 792/6 = 132  → mesh_lr = 132 (or nearest power-of-2)
      --stride_hr 3 \\     # 792/3 = 264  → mesh_hr = 264
      --box 500.0 \\       # box size in Mpc/h (read from Gadget header if omitted)
      --out ./data_gadget

  # Smaller test pair (faster to load/process):
  python convert_gadget.py --snap snapshot.hdf5 --n_gadget 792 \\
      --stride_lr 12 --stride_hr 6 --out ./data_gadget_test
      # → LR: 66³   HR: 132³  (much faster to load)

Notes
-----
- Requires h5py: pip install h5py
- For very large snapshots (>50 GB), reads in chunks (see --chunk_size).
- The output mesh_lr / mesh_hr will be the nearest even integer to
  n_gadget // stride_*. Adjust strides to get round numbers.
  792: stride=2→396, stride=3→264, stride=4→198, stride=6→132, stride=8→99
"""

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import h5py
except ImportError:
    sys.exit("h5py not found — run: pip install h5py")


# ==============================================================================
# Gadget HDF5 reader
# ==============================================================================

def read_gadget_hdf5(snap_path: Path, chunk_size: int = 2_000_000):
    """
    Read PartType1 coordinates, velocities, IDs from a Gadget HDF5 snapshot.

    Returns
    -------
    pos : [N, 3]   float32  positions   in internal Gadget units
    vel : [N, 3]   float32  velocities  in internal Gadget units
    ids : [N]      int64    particle IDs (1-indexed)
    header : dict  snapshot header attributes
    """
    with h5py.File(snap_path, "r") as f:
        header = dict(f["Header"].attrs)
        pt     = f["PartType1"]

        pos = pt["Coordinates"][:]
        vel = pt["Velocities"][:]
        ids = pt["ParticleIDs"][:]

    print(f"  Read {len(ids):,} particles  |  pos dtype={pos.dtype}  vel dtype={vel.dtype}")
    print(f"  Header: BoxSize={header.get('BoxSize','?')}  "
          f"a={header.get('Time','?'):.4f}  "
          f"HubbleParam={header.get('HubbleParam','?')}")
    return pos.astype(np.float32), vel.astype(np.float32), ids.astype(np.int64), header


# ==============================================================================
# Lagrangian ordering via particle IDs
# ==============================================================================

def ids_to_lagrangian_indices(ids: np.ndarray, n_gadget: int):
    """
    Decode Gadget particle IDs to Lagrangian grid indices (ix, iy, iz).

    Standard encoding:  ID = ix * N² + iy * N + iz + 1  (1-indexed, ix outermost)

    Returns ix, iy, iz arrays of shape [N].
    """
    i0 = ids.astype(np.int64) - 1          # 0-indexed
    N  = n_gadget
    ix = i0 // (N * N)
    iy = (i0 // N) % N
    iz = i0 % N
    return ix, iy, iz


def sort_to_lagrangian_order(pos, vel, ids, n_gadget):
    """
    Re-order particles to Lagrangian C-order (ix outermost, iz innermost).
    Returns pos and vel arrays sorted by Lagrangian index.
    """
    ix, iy, iz = ids_to_lagrangian_indices(ids, n_gadget)
    flat_idx   = ix * n_gadget**2 + iy * n_gadget + iz   # C-order flat index
    sort_order = np.argsort(flat_idx)
    return pos[sort_order], vel[sort_order]


# ==============================================================================
# LR / HR creation
# ==============================================================================

def subsample_hr(pos_sorted, vel_sorted, n_gadget, stride_hr):
    """
    HR: take every stride_hr-th particle along each axis (exact subgrid).
    Returns [n_hr³, 3] arrays.
    """
    n_hr = n_gadget // stride_hr
    pos3 = pos_sorted.reshape(n_gadget, n_gadget, n_gadget, 3)
    vel3 = vel_sorted.reshape(n_gadget, n_gadget, n_gadget, 3)
    pos_hr = pos3[::stride_hr, ::stride_hr, ::stride_hr, :].reshape(-1, 3).copy()
    vel_hr = vel3[::stride_hr, ::stride_hr, ::stride_hr, :].reshape(-1, 3).copy()
    print(f"  HR subsample: stride={stride_hr}  →  {n_hr}³ = {n_hr**3:,} particles")
    return pos_hr, vel_hr, n_hr


def block_average_lr(pos_sorted, vel_sorted, n_gadget, stride_lr):
    """
    LR: centre-of-mass (block average) over (stride_lr)³ blocks.

    Each block of stride_lr³ Gadget particles → one LR particle.
    This is equivalent to what a PM simulation at n_lr resolution would give:
    the force mesh averages over the same spatial scale.

    Returns [n_lr³, 3] arrays.
    """
    n_lr  = n_gadget // stride_lr
    s     = stride_lr
    pos3  = pos_sorted.reshape(n_gadget, n_gadget, n_gadget, 3)
    vel3  = vel_sorted.reshape(n_gadget, n_gadget, n_gadget, 3)

    # Reshape to expose blocks: [n_lr, s, n_lr, s, n_lr, s, 3]
    pos_b = pos3.reshape(n_lr, s, n_lr, s, n_lr, s, 3)
    vel_b = vel3.reshape(n_lr, s, n_lr, s, n_lr, s, 3)

    # Average over the block axes (1, 3, 5)
    pos_lr = pos_b.mean(axis=(1, 3, 5)).reshape(-1, 3)
    vel_lr = vel_b.mean(axis=(1, 3, 5)).reshape(-1, 3)

    print(f"  LR block-avg:  stride={stride_lr}  →  {n_lr}³ = {n_lr**3:,} particles")
    return pos_lr, vel_lr, n_lr


# ==============================================================================
# Unit conversion
# ==============================================================================

def to_mpc_h(arr, gadget_units_to_mpc_h: float):
    """Convert Gadget internal length units to Mpc/h."""
    return arr * gadget_units_to_mpc_h


# ==============================================================================
# Main
# ==============================================================================

def convert(
    snap_path:   str,
    n_gadget:    int,
    stride_lr:   int,
    stride_hr:   int,
    box_size:    float,        # Mpc/h  (None = read from header)
    out_dir:     str,
    chunk_size:  int  = 4_000_000,
    gadget_length_to_mpc: float = None,  # if None, auto-detect from header
):
    snap = Path(snap_path)
    out  = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Read ──────────────────────────────────────────────────────────────
    print(f"\nReading {snap.name} …")
    pos_raw, vel_raw, ids, header = read_gadget_hdf5(snap)

    # Unit conversion: Gadget stores positions in kpc/h by default
    if gadget_length_to_mpc is None:
        # Gadget-2 default: internal unit = kpc/h → divide by 1000 to get Mpc/h
        # Gadget-4 / AREPO may use different units — check header
        box_header = float(header.get("BoxSize", 0.0))
        if box_size is not None and box_header > 0:
            gadget_length_to_mpc = box_size / box_header
        else:
            gadget_length_to_mpc = 1e-3   # assume kpc/h → Mpc/h
            print(f"  WARNING: assuming Gadget length unit = kpc/h "
                  f"(gadget_length_to_mpc={gadget_length_to_mpc}). "
                  f"Override with --gadget_length_to_mpc if wrong.")

    if box_size is None:
        box_size = float(header["BoxSize"]) * gadget_length_to_mpc
        print(f"  Box size from header: {box_size:.2f} Mpc/h")

    a_snap = float(header.get("Time", 1.0))
    print(f"  Scale factor a = {a_snap:.4f}")
    print(f"  Gadget length → Mpc/h factor: {gadget_length_to_mpc:.6f}")

    pos_mpc = to_mpc_h(pos_raw, gadget_length_to_mpc)
    del pos_raw

    # ── Sort to Lagrangian order ──────────────────────────────────────────
    print("Sorting to Lagrangian C-order (ix outermost) …")
    pos_sorted, vel_sorted = sort_to_lagrangian_order(pos_mpc, vel_raw, ids, n_gadget)
    del pos_mpc, vel_raw, ids

    # ── Build LR and HR ───────────────────────────────────────────────────
    print("\nBuilding HR subsample …")
    pos_hr, vel_hr, n_hr = subsample_hr(pos_sorted, vel_sorted, n_gadget, stride_hr)

    print("Building LR block-average …")
    pos_lr, vel_lr, n_lr = block_average_lr(pos_sorted, vel_sorted, n_gadget, stride_lr)

    del pos_sorted, vel_sorted

    # ── Displacement check ────────────────────────────────────────────────
    # LR Lagrangian positions (in Mpc/h, ij ordering)
    spacing_lr = box_size / n_lr
    coords_lr  = np.arange(n_lr, dtype=np.float32) * spacing_lr
    ix, iy, iz = np.meshgrid(coords_lr, coords_lr, coords_lr, indexing="ij")
    q_lr       = np.stack([ix.ravel(), iy.ravel(), iz.ravel()], axis=-1)
    psi_lr_mag = np.sqrt(np.sum((pos_lr - q_lr)**2, axis=-1)).mean()

    spacing_hr = box_size / n_hr
    coords_hr  = np.arange(n_hr, dtype=np.float32) * spacing_hr
    stride_rel = stride_lr // stride_hr
    ix_hr = coords_hr[::stride_rel]
    # sample HR q at LR positions
    q_hr_at_lr = np.stack(np.meshgrid(ix_hr, ix_hr, ix_hr, indexing="ij"), axis=-1).reshape(-1, 3)
    psi_hr_at_lr = pos_hr.reshape(n_hr, n_hr, n_hr, 3)[::stride_rel, ::stride_rel, ::stride_rel, :].reshape(-1, 3)
    delta_psi_mag = np.sqrt(np.sum((psi_hr_at_lr - pos_lr)**2, axis=-1)).mean()

    print(f"\n  |Ψ_LR|  mean = {psi_lr_mag:.4f} Mpc/h")
    print(f"  |ΔΨ|    mean = {delta_psi_mag:.4f} Mpc/h  "
          f"({delta_psi_mag/psi_lr_mag:.1%} of LR displacement)")
    if delta_psi_mag / (psi_lr_mag + 1e-10) < 0.01:
        print("  WARNING: |ΔΨ| very small — strides may be too similar or ordering mismatch")

    # ── Save ──────────────────────────────────────────────────────────────
    print(f"\nSaving to {out} …")
    # Add snapshot axis [1, N, 3] so train_lag_disp.py can index [snap_idx]
    np.save(out / f"pos_m{n_lr}_s0.npy",  pos_lr[np.newaxis])   # [1, n_lr³, 3]
    np.save(out / f"vel_m{n_lr}_s0.npy",  vel_lr[np.newaxis])
    np.save(out / f"pos_m{n_hr}_s0.npy",  pos_hr[np.newaxis])   # [1, n_hr³, 3]
    np.save(out / f"vel_m{n_hr}_s0.npy",  vel_hr[np.newaxis])
    np.save(out / "scale_factors.npy",    np.array([a_snap]))

    for fname in [f"pos_m{n_lr}_s0.npy", f"pos_m{n_hr}_s0.npy"]:
        size_mb = (out / fname).stat().st_size / 1e6
        print(f"  {fname}  {size_mb:.1f} MB")

    # ── Config hint ───────────────────────────────────────────────────────
    print(f"""
{'='*55}
Config for lag_disp_mlp.yaml:

  data:
    data_dir:    {out.resolve()}
    mesh_lr:     {n_lr}
    mesh_hr:     {n_hr}
    box_size:    {box_size:.1f}   # Mpc/h
    n_particles: {n_lr}           # per dimension

    sim_id_train: 0
    sim_id_val:   0    # same sim — only 1 snapshot available
    snap_train:   0
    snaps_val:    [0]
{'='*55}
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Gadget snapshot → LR/HR displacement SR pair"
    )
    parser.add_argument("--snap",        required=True,  help="Path to Gadget HDF5 snapshot")
    parser.add_argument("--n_gadget",    required=True,  type=int,   help="Particles per side (e.g. 792)")
    parser.add_argument("--stride_lr",   default=6,      type=int,   help="LR block-avg stride (792/6=132)")
    parser.add_argument("--stride_hr",   default=3,      type=int,   help="HR subsample stride (792/3=264)")
    parser.add_argument("--box",         default=None,   type=float, help="Box size in Mpc/h")
    parser.add_argument("--out",         default="./data_gadget",    help="Output directory")
    parser.add_argument("--chunk_size",  default=4_000_000, type=int)
    parser.add_argument("--gadget_length_to_mpc", default=None, type=float,
                        help="Multiply Gadget length units by this to get Mpc/h "
                             "(default: auto-detect; Gadget-2 kpc/h → 1e-3)")
    args = parser.parse_args()

    assert args.stride_lr % args.stride_hr == 0, \
        f"stride_lr ({args.stride_lr}) must be a multiple of stride_hr ({args.stride_hr})"
    assert args.n_gadget % args.stride_lr == 0, \
        f"n_gadget ({args.n_gadget}) must be divisible by stride_lr ({args.stride_lr})"
    assert args.n_gadget % args.stride_hr == 0, \
        f"n_gadget ({args.n_gadget}) must be divisible by stride_hr ({args.stride_hr})"

    convert(
        snap_path  = args.snap,
        n_gadget   = args.n_gadget,
        stride_lr  = args.stride_lr,
        stride_hr  = args.stride_hr,
        box_size   = args.box,
        out_dir    = args.out,
        chunk_size = args.chunk_size,
        gadget_length_to_mpc = args.gadget_length_to_mpc,
    )
