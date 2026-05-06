"""
lagrangian.py — Lagrangian feature extraction for PM force-correction experiments.

Physical motivation
-------------------
Particles are labelled by their initial (Lagrangian) grid index q_i.  As the
simulation evolves, the local mapping q → x(t) is encoded by the DEFORMATION
TENSOR at particle i:

    D[i, α, β]  =  ∂x^α_i / ∂q^β
                ≈  (x^α_i(t) − x^α_{j,−β}(t)) / Δq

where j,−β is the Lagrangian neighbour of i in the −β direction and
Δq = mesh_lr / n_part is the initial lattice spacing.

Key derived quantities:
  D ≈ I     at initial conditions (no displacement)
  E = D − I  strain tensor (zero at IC, grows with structure formation)
  det(D) < 0  shell-crossing — the local Lagrangian patch has been inverted

The STRAIN tensor E encodes:
  tr(E)       volume change → local overdensity (partially redundant with CNN δ)
  det(D)      shell-crossing signal → INVISIBLE to Eulerian field
  eigenvalues of D → anisotropic collapse (filaments, sheets, halos)

The Eulerian density field is DEGENERATE over multi-stream configurations
(multiple particles at the same Eulerian position from different Lagrangian
origins).  The deformation tensor breaks this degeneracy.

Module structure
----------------
  get_axis_neighbor_indices(n_part)             — static precomputed [N, 6]
  lagrangian_lattice_positions(n_part, mesh_lr) — exact initial grid q_i [N, 3]
  compute_deformation_features(...)             — [N, D_feat] per snapshot
  LagrangianForceCorrector                      — Haiku: features → ΔF [N,3] or ΔΦ [N,1]
  make_lagrangian_corrector(config)             — factory that returns hk model
"""

import jax
import jax.numpy as jnp
import haiku as hk
from typing import Optional


# ==============================================================================
# 1. Particle ordering and neighbour indices
# ==============================================================================

def get_axis_neighbor_indices(n_part: int) -> jnp.ndarray:
    """
    Precompute 6 axis-aligned Lagrangian neighbour indices for all particles.

    Particles are assumed in C-order: i = ix*n² + iy*n + iz
    with periodic boundary conditions.

    Returns
    -------
    idx : [N, 6]  int32
        Columns ordered: +x, −x, +y, −y, +z, −z
        The −direction columns (1, 3, 5) are used for deformation tensor;
        all 6 are exposed for future K-neighbour extensions.
    """
    n = n_part
    N = n * n * n
    i = jnp.arange(N)

    ix = i // (n * n)
    iy = (i // n) % n
    iz = i % n

    def flat(x, y, z):
        return (x % n) * n * n + (y % n) * n + (z % n)

    return jnp.stack([
        flat(ix + 1, iy,     iz    ),   # 0: +x
        flat(ix - 1, iy,     iz    ),   # 1: −x
        flat(ix,     iy + 1, iz    ),   # 2: +y
        flat(ix,     iy - 1, iz    ),   # 3: −y
        flat(ix,     iy,     iz + 1),   # 4: +z
        flat(ix,     iy,     iz - 1),   # 5: −z
    ], axis=-1).astype(jnp.int32)       # [N, 6]


def lagrangian_lattice_positions(n_part: int, mesh_lr: int) -> jnp.ndarray:
    """
    Exact initial lattice positions q_i in mesh units [0, mesh_lr).

    Cell centres at (ix + 0.5) * Δq where Δq = mesh_lr / n_part.
    Ordering: C-order (i = ix*n² + iy*n + iz).
    """
    n  = n_part
    N  = n * n * n
    dq = mesh_lr / n

    i  = jnp.arange(N)
    ix = i // (n * n)
    iy = (i // n) % n
    iz = i % n

    q  = jnp.stack([ix, iy, iz], axis=-1).astype(jnp.float32)
    return (q + 0.5) * dq   # [N, 3]  cell centres in mesh units


# ==============================================================================
# 2. Deformation tensor computation
# ==============================================================================

def compute_deformation_tensor(
    pos_t: jnp.ndarray,
    neg_neighbor_idx: jnp.ndarray,
    mesh_lr: int,
) -> tuple:
    """
    Compute per-particle deformation tensor D ∈ ℝ^{N×3×3} from the
    three backward-axis (−x, −y, −z) Lagrangian neighbours.

    D[i, α, β] = (x^α_i − x^α_{j,−β}) / Δq   (finite-difference ∂x^α/∂q^β)

    Uses periodic minimum-image convention for differences.

    Parameters
    ----------
    pos_t           : [N, 3]   current positions in mesh_lr units
    neg_neighbor_idx: [N, 3]   indices of −x, −y, −z neighbours
    mesh_lr         : int      grid size (for Δq and periodic wrap)

    Returns
    -------
    D_flat : [N, 9]   row-major flattened deformation tensor  (α outer, β inner)
    det_D  : [N]      Jacobian determinant  (< 0 → shell-crossing)
    """
    n_part  = round(pos_t.shape[0] ** (1.0 / 3.0))
    delta_q = mesh_lr / n_part   # initial lattice spacing in mesh units

    # x_neg[i, β, α] = position of the −β neighbour of particle i, component α
    x_neg = pos_t[neg_neighbor_idx]         # [N, 3, 3]

    # diff[i, β, α] = x^α_i − x^α_{j,−β}
    diff  = pos_t[:, None, :] - x_neg       # [N, 3, 3]
    diff  = diff - mesh_lr * jnp.round(diff / mesh_lr)   # periodic min-image

    # Normalize → D[i, β, α] = diff / Δq
    D = diff / delta_q                       # [N, 3, 3]  (β outer, α inner)

    # Transpose to standard convention D[i, α, β] = ∂x^α/∂q^β
    D = jnp.transpose(D, (0, 2, 1))         # [N, 3, 3]  (α outer, β inner)

    det_D  = jnp.linalg.det(D)              # [N]
    D_flat = D.reshape(D.shape[0], 9)        # [N, 9]

    return D_flat, det_D


# ==============================================================================
# 3. Feature assembly
# ==============================================================================

def compute_deformation_features(
    pos_t: jnp.ndarray,
    neighbor_idx: jnp.ndarray,
    mesh_lr: int,
    use_strain: bool = True,
    use_invariants: bool = False,
) -> jnp.ndarray:
    """
    Assemble per-particle Lagrangian feature vector from axis-aligned neighbours.

    Parameters
    ----------
    pos_t          : [N, 3]   current positions (mesh_lr units)
    neighbor_idx   : [N, 6]   precomputed axis-neighbour indices (+x,−x,+y,−y,+z,−z)
    mesh_lr        : int
    use_strain     : if True return STRAIN E = D − I (zero at IC, clean signal)
                     if False return raw deformation D
    use_invariants : if True append [tr(D), det(D), ||E||_F] — 3 extra features
                     that are physically interpretable and coordinate-independent

    Returns
    -------
    feats : [N, 9] or [N, 12]
    """
    neg_idx        = neighbor_idx[:, [1, 3, 5]]          # −x, −y, −z
    D_flat, det_D  = compute_deformation_tensor(pos_t, neg_idx, mesh_lr)

    D_3x3 = D_flat.reshape(-1, 3, 3)
    I3    = jnp.eye(3, dtype=jnp.float32)[None]          # [1, 3, 3]
    E     = D_3x3 - I3                                    # strain

    feats = E.reshape(-1, 9) if use_strain else D_flat

    if use_invariants:
        tr_D  = jnp.trace(D_3x3, axis1=1, axis2=2)           # [N]  volume change
        # det_D already computed
        fro_E = jnp.sqrt(jnp.sum(E ** 2, axis=(1, 2)) + 1e-12)  # [N]  total strain
        extra = jnp.stack([tr_D, det_D, fro_E], axis=-1)      # [N, 3]
        feats = jnp.concatenate([feats, extra], axis=-1)

    return feats   # [N, 9] or [N, 12]


# ==============================================================================
# 4. Haiku model
# ==============================================================================

class LagrangianForceCorrector(hk.Module):
    """
    Per-particle MLP that maps Lagrangian deformation features to a force
    correction (vector) or potential correction (scalar).

    Input per particle
    ------------------
    deform_feats : [N, D_feat]  strain / deformation features (from compute_deformation_features)
    velocities   : [N, 3]       current velocities (context)
    scale_factors: scalar       scale factor a

    Output
    ------
    [N, output_dim]   where output_dim=3 for direct force, 1 for scalar potential

    Notes
    -----
    * SiLU (Swish) activations for smooth force predictions.
    * Linear output layer (no activation) — correct for regression.
    * Designed to be used as a standalone model OR combined with CNN-WST
      (the caller handles the summation of corrections).
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 3,
        output_dim: int = 3,   # 3 = force,  1 = potential
        name: Optional[str] = None,
    ):
        super().__init__(name=name or "LagrangianForceCorrector")
        self.hidden_dim = hidden_dim
        self.n_layers   = n_layers
        self.output_dim = output_dim

    def __call__(
        self,
        deform_feats: jnp.ndarray,
        velocities: jnp.ndarray,
        scale_factors,
    ) -> jnp.ndarray:
        N     = deform_feats.shape[0]
        a_col = jnp.ones((N, 1), dtype=jnp.float32) * scale_factors

        x = jnp.concatenate([deform_feats, velocities, a_col], axis=-1)

        output_sizes = [self.hidden_dim] * self.n_layers + [self.output_dim]
        return hk.nets.MLP(
            output_sizes=output_sizes,
            activation=jax.nn.silu,   # smooth → better force gradients
            name="lag_mlp",
        )(x)   # [N, output_dim]


# ==============================================================================
# 5. Factory
# ==============================================================================

def make_lagrangian_corrector(
    hidden_dim: int = 64,
    n_layers: int = 3,
    output_dim: int = 3,
):
    """
    Returns hk.without_apply_rng(hk.transform(LagrangianCorr)).

    Usage
    -----
        model  = make_lagrangian_corrector(hidden_dim=64, n_layers=3, output_dim=3)
        params = model.init(rng, deform_feats, velocities, scale_factor)
        delta_f = model.apply(params, deform_feats, velocities, a)  # [N, 3] or [N, 1]
    """
    def LagrangianCorr(deform_feats, velocities, scale_factors):
        return LagrangianForceCorrector(
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            output_dim=output_dim,
        )(deform_feats, velocities, scale_factors)

    return hk.without_apply_rng(hk.transform(LagrangianCorr))
