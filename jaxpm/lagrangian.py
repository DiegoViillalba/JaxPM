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

Extended neighbourhood (n_shell > 0)
-------------------------------------
The deformation tensor D is a FIRST-ORDER approximation of the local flow.
It captures the average strain at particle i but misses:
  • How rapidly the deformation changes across the neighbourhood (Lagrangian Hessian)
  • Asymmetry between + and − directions (non-linear deformation)
  • Multi-stream structure: two particles with the same D can sit in very
    different environments if their neighbours have diverged asymmetrically.

The extended features address this by computing, for each of the K neighbours
in a cubic shell of radius n_shell (K = (2·n_shell+1)³ − 1):

    δe_k = (x_k − x_i) / Δq − D_i · Δq_k

where Δq_k ∈ ℤ³ is the LAGRANGIAN OFFSET (in lattice units).
The term D_i · Δq_k is the LINEAR PREDICTION of neighbour k's displacement
given the local deformation tensor.  The residual δe_k measures how much
the actual flow DEVIATES from linear — it is zero for perfectly linear
flows and large near shell-crossing, halos, and multi-streaming regions.

These residuals are pooled over all K neighbours:
  pool_mode = "mean"       → 3 features  (mean residual vector)
  pool_mode = "var"        → 3 features  (mean squared residual per component)
  pool_mode = "mean_var"   → 6 features  (both; default)

Shell sizes:
  n_shell=0 → only D tensor (current behaviour)   K=0
  n_shell=1 → 3³−1 = 26 neighbours               K=26
  n_shell=2 → 5³−1 = 124 neighbours              K=124

Feature dimension table (use_strain=True):
  n_shell=0, use_invariants=False  →  9
  n_shell=0, use_invariants=True   → 12
  n_shell=1, use_invariants=False  → 15  (9 + 6 env)
  n_shell=1, use_invariants=True   → 18
  n_shell=2, use_invariants=False  → 15  (same pooled dim regardless of K)
  n_shell=2, use_invariants=True   → 18

Time conditioning (FiLM)
------------------------
The scale factor a encodes the COSMIC TIME at which the snapshot is taken.
Simple concatenation to the input only lets the first layer see a; FiLM
(Feature-wise Linear Modulation, Perez et al. 2018) does better:

    emb_a  = TimeEmbedding(a)          [D_e]  sinusoidal + MLP
    γ_l, β_l = film_proj_l(emb_a)      [D_h] each — per-layer scale/shift
    h_l    = (1 + γ_l) ⊙ fc_l(h_{l-1}) + β_l

Every hidden layer is modulated by a, so the model can learn DIFFERENT
force-displacement relations at different epochs (recollapse, shell-crossing
onset, virialization) without sharing a single "time-agnostic" representation.

Zero-init of film projections → at step 0 the network is a plain MLP,
so training is stable regardless of the conditioning strength.

Module structure
----------------
  get_axis_neighbor_indices(n_part)                — static [N, 6] (backward-compat)
  get_shell_neighbor_indices(n_part, n_shell)      — static [N, K] + offsets + slices
  lagrangian_lattice_positions(n_part, mesh_lr)    — exact initial grid q_i [N, 3]
  compute_deformation_tensor(...)                  — [N, 9] + det [N]
  compute_environment_features(...)                — [N, 3|6] non-linear residuals
  compute_deformation_features(...)                — [N, D_feat] full feature assembly
  TimeEmbedding                                    — Haiku: scalar a → [D_e]
  LagrangianForceCorrector                         — Haiku: features → ΔF [N,3] or ΔΦ [N,1]
  make_lagrangian_corrector(config)                — factory that returns hk model
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


def get_shell_neighbor_indices(
    n_part: int,
    n_shell: int = 1,
) -> tuple:
    """
    Precompute Lagrangian neighbour indices for all particles within a cubic
    shell of radius n_shell.

    Covers every offset (dx, dy, dz) ∈ {−n_shell … n_shell}³ excluding (0,0,0),
    giving K = (2·n_shell+1)³ − 1 neighbours.

    Offsets are sorted by squared Lagrangian distance so that face neighbours
    (dist²=1) come first, then edge (dist²=2), corner (dist²=3), etc.  This
    ordering lets callers extract per-shell slices by index without any dynamic
    masking (useful inside jit).

    Parameters
    ----------
    n_part  : int   particles per dimension
    n_shell : int   shell radius in lattice units (1 → 26 nbrs, 2 → 124 nbrs)

    Returns
    -------
    idx     : [N, K]  int32   global particle indices of all neighbours
    offsets : [K, 3]  int32   Lagrangian offsets (dx, dy, dz) in lattice units,
                              sorted by dist² = dx²+dy²+dz²
    shell_slices : tuple of (start, end) pairs, one per distinct dist² value,
                   ordered by increasing dist².
                   Example for n_shell=1: ((0, 6), (6, 18), (18, 26))
                   These are STATIC Python ints — safe to use as jit array slices.
    """
    # ── Build sorted offset list ──────────────────────────────────────────
    offsets_list = [
        (dx, dy, dz)
        for dx in range(-n_shell, n_shell + 1)
        for dy in range(-n_shell, n_shell + 1)
        for dz in range(-n_shell, n_shell + 1)
        if not (dx == 0 and dy == 0 and dz == 0)
    ]
    offsets_list.sort(key=lambda o: o[0] ** 2 + o[1] ** 2 + o[2] ** 2)

    # ── Compute shell_slices (static Python ints) ─────────────────────────
    shell_slices = []
    cur_d2, start = None, 0
    for k, (dx, dy, dz) in enumerate(offsets_list):
        d2 = dx ** 2 + dy ** 2 + dz ** 2
        if d2 != cur_d2:
            if cur_d2 is not None:
                shell_slices.append((start, k))
            cur_d2, start = d2, k
    if cur_d2 is not None:
        shell_slices.append((start, len(offsets_list)))
    shell_slices = tuple(shell_slices)

    offsets = jnp.array(offsets_list, dtype=jnp.int32)   # [K, 3]

    # ── Build [N, K] global index array ──────────────────────────────────
    n = n_part
    N = n * n * n
    i  = jnp.arange(N)
    ix = i // (n * n)
    iy = (i // n) % n
    iz = i % n

    def flat(x, y, z):
        return (x % n) * n * n + (y % n) * n + (z % n)

    cols = [flat(ix + dx, iy + dy, iz + dz) for dx, dy, dz in offsets_list]
    idx  = jnp.stack(cols, axis=-1).astype(jnp.int32)    # [N, K]

    return idx, offsets, shell_slices


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
# 3. Extended neighbourhood features
# ==============================================================================

def compute_environment_features(
    pos_t: jnp.ndarray,
    D_flat: jnp.ndarray,
    ext_neighbor_idx: jnp.ndarray,
    ext_offsets: jnp.ndarray,
    mesh_lr: int,
    pool_mode: str = "mean_var",
    shell_slices: tuple = (),
) -> jnp.ndarray:
    """
    Per-particle non-linear environment features from an extended neighbourhood.

    For each of K neighbours k at Lagrangian offset Δq_k (lattice units):

        δe_k = (x_k − x_i) / Δq  −  D_i · Δq_k

    The second term is the LINEAR PREDICTION of where neighbour k should be
    given the local deformation tensor D_i.  The residual δe_k ∈ ℝ³ is:
      • zero for perfectly linear flows
      • non-zero where the deformation ACCELERATES or CURVES between particles
      • largest near shell-crossing, halos, and multi-streaming regions

    These residuals are pooled over all K neighbours to produce a fixed-size
    feature vector regardless of K (i.e., the number of neighbours).

    Parameters
    ----------
    pos_t            : [N, 3]  current positions in mesh_lr units
    D_flat           : [N, 9]  flattened deformation tensor (from compute_deformation_tensor)
    ext_neighbor_idx : [N, K]  global indices of the K neighbours
    ext_offsets      : [K, 3]  int32 Lagrangian offsets (dx, dy, dz) in lattice units
    mesh_lr          : int
    pool_mode        : "mean"       → [N, 3]  mean residual vector
                       "var"        → [N, 3]  mean squared residual (per component)
                       "mean_var"   → [N, 6]  both concatenated  (default)
                       "shell_mean" → [N, 3·S]  mean per shell type (requires shell_slices)
                       "shell_mean_var" → [N, 6·S]  mean+var per shell type
    shell_slices     : tuple of (start, end) ints — from get_shell_neighbor_indices.
                       Required only for "shell_mean" and "shell_mean_var" modes.
                       Must be static (Python ints) to be jit-safe.

    Returns
    -------
    env_feats : [N, n_env_feats]
        pool_mode     n_env_feats
        "mean"              3
        "var"               3
        "mean_var"          6
        "shell_mean"      3·S  (S = number of shells = len(shell_slices))
        "shell_mean_var"  6·S
    """
    n_part  = round(pos_t.shape[0] ** (1.0 / 3.0))
    delta_q = float(mesh_lr) / float(n_part)   # lattice spacing in mesh units

    # ── Gather neighbour positions ────────────────────────────────────────
    x_nbrs = pos_t[ext_neighbor_idx]            # [N, K, 3]

    # ── Displacement from particle i to each neighbour (periodic min-image)
    diffs = x_nbrs - pos_t[:, None, :]          # [N, K, 3]
    diffs = diffs - mesh_lr * jnp.round(diffs / mesh_lr)
    diffs_normed = diffs / delta_q               # [N, K, 3]  — in lattice units

    # ── Linear prediction: D_i · offset_k ────────────────────────────────
    # D_flat[i].reshape(3,3) @ offsets[k] = linear displacement prediction
    D_3x3    = D_flat.reshape(-1, 3, 3)          # [N, 3, 3]
    off_f    = ext_offsets.astype(jnp.float32)   # [K, 3]
    # lin[i, k, α] = Σ_β D_i[α, β] · off_f[k, β]
    lin_pred = jnp.einsum("nab,kb->nka", D_3x3, off_f)  # [N, K, 3]

    # ── Non-linear residuals ──────────────────────────────────────────────
    residuals = diffs_normed - lin_pred           # [N, K, 3]

    if pool_mode == "mean":
        return residuals.mean(axis=1)             # [N, 3]

    if pool_mode == "var":
        return (residuals ** 2).mean(axis=1)      # [N, 3]

    if pool_mode == "mean_var":
        return jnp.concatenate([
            residuals.mean(axis=1),               # [N, 3]
            (residuals ** 2).mean(axis=1),         # [N, 3]
        ], axis=-1)                                # [N, 6]

    if pool_mode in ("shell_mean", "shell_mean_var") and shell_slices:
        # shell_slices is a static tuple of (start, end) Python ints —
        # slicing with static ints is always jit-safe.
        parts = []
        for (s, e) in shell_slices:
            slab = residuals[:, s:e, :]            # [N, shell_K, 3]
            parts.append(slab.mean(axis=1))        # [N, 3]
            if pool_mode == "shell_mean_var":
                parts.append((slab ** 2).mean(axis=1))  # [N, 3]
        return jnp.concatenate(parts, axis=-1)

    raise ValueError(
        f"Unknown pool_mode '{pool_mode}'. "
        f"Choose: 'mean', 'var', 'mean_var', 'shell_mean', 'shell_mean_var'."
    )


# ==============================================================================
# 4. Feature assembly
# ==============================================================================

def compute_deformation_features(
    pos_t: jnp.ndarray,
    neighbor_idx: jnp.ndarray,
    mesh_lr: int,
    use_strain: bool = True,
    use_invariants: bool = False,
    ext_neighbor_idx=None,
    ext_offsets=None,
    pool_mode: str = "mean_var",
    shell_slices: tuple = (),
) -> jnp.ndarray:
    """
    Assemble per-particle Lagrangian feature vector.

    Core features (always computed)
    --------------------------------
    Uses the 3 backward-axis (−x,−y,−z) neighbours to compute the deformation
    tensor D[i,α,β] = ∂x^α/∂q^β and derive:
      • E = D − I  (strain, 9 components)  if use_strain=True
      • D_flat      (raw deformation, 9)    if use_strain=False
      • [tr D, det D, ‖E‖_F]  (3 invariants)  if use_invariants=True

    Extended neighbourhood (optional, n_shell > 0)
    ------------------------------------------------
    When ext_neighbor_idx and ext_offsets are provided (from
    get_shell_neighbor_indices), non-linear environment features are appended:

        δe_k = (x_k − x_i)/Δq − D_i · Δq_k   for each of K neighbours k

    These residuals capture how much the flow deviates from the linear
    prediction — zero for laminar flows, large near shell-crossing and halos.
    They are pooled into a fixed-size vector via pool_mode.

    Parameters
    ----------
    pos_t            : [N, 3]   current positions (mesh_lr units)
    neighbor_idx     : [N, 6]   axis-neighbour indices (+x,−x,+y,−y,+z,−z)
    mesh_lr          : int
    use_strain       : True → return E = D−I  (zero at IC, cleaner signal)
    use_invariants   : True → append [tr D, det D, ‖E‖_F]
    ext_neighbor_idx : [N, K] int32  — from get_shell_neighbor_indices; or None
    ext_offsets      : [K, 3] int32  — from get_shell_neighbor_indices; or None
    pool_mode        : how to pool K environment residuals (see compute_environment_features)
    shell_slices     : static tuple of (start,end) ints for shell_mean modes

    Returns
    -------
    feats : [N, D_feat]
        D_feat depends on flags:
          use_strain  use_invariants  ext (mean_var)
            T             F              N   →   9
            T             T              N   →  12
            T             F              Y   →  15
            T             T              Y   →  18
    """
    neg_idx       = neighbor_idx[:, [1, 3, 5]]          # −x, −y, −z
    D_flat, det_D = compute_deformation_tensor(pos_t, neg_idx, mesh_lr)

    D_3x3 = D_flat.reshape(-1, 3, 3)
    I3    = jnp.eye(3, dtype=jnp.float32)[None]         # [1, 3, 3]
    E     = D_3x3 - I3                                   # strain

    feats = E.reshape(-1, 9) if use_strain else D_flat   # [N, 9]

    if use_invariants:
        tr_D  = jnp.trace(D_3x3, axis1=1, axis2=2)          # [N]
        fro_E = jnp.sqrt(jnp.sum(E ** 2, axis=(1, 2)) + 1e-12)
        extra = jnp.stack([tr_D, det_D, fro_E], axis=-1)    # [N, 3]
        feats = jnp.concatenate([feats, extra], axis=-1)     # [N, 12]

    # ── Extended neighbourhood ────────────────────────────────────────────
    if ext_neighbor_idx is not None and ext_offsets is not None:
        env = compute_environment_features(
            pos_t, D_flat,
            ext_neighbor_idx, ext_offsets,
            mesh_lr, pool_mode, shell_slices,
        )                                                    # [N, n_env]
        feats = jnp.concatenate([feats, env], axis=-1)

    return feats


# ==============================================================================
# 4. Time embedding (sinusoidal + MLP projection)
# ==============================================================================

class TimeEmbedding(hk.Module):
    """
    Maps the scalar scale factor a ∈ (0, 1] to a rich fixed-size vector.

    Architecture
    ------------
    1. Sinusoidal Fourier features:
           freqs  = exp(−log(F_max) · k / (K−1))   k = 0 … K−1
           sincos = [sin(a·freqs), cos(a·freqs)]    ∈ ℝ^{2K}
       These are smooth, non-saturating, and capture multiple time-scales
       simultaneously (fast oscillations at early times, slow at late times).

    2. Two-layer MLP projection sincos → emb_a ∈ ℝ^{embed_dim}
       with SiLU activation → learns a non-linear time manifold.

    Parameters
    ----------
    embed_dim : int     output dimension (default 32)
    freq_max  : float   maximum frequency (default 1000.0)

    Notes
    -----
    * embed_dim should be even (half used for sin, half for cos internally).
    * The projection MLP uses 2×embed_dim hidden units.
    """

    def __init__(
        self,
        embed_dim: int = 32,
        freq_max: float = 1000.0,
        name: Optional[str] = None,
    ):
        super().__init__(name=name or "TimeEmbedding")
        self.embed_dim = embed_dim
        self.freq_max  = freq_max

    def __call__(self, a) -> jnp.ndarray:
        """
        Parameters
        ----------
        a : scalar  (Python float or 0-d JAX array)

        Returns
        -------
        emb : [embed_dim]
        """
        half   = self.embed_dim // 2
        # Geometric sequence of frequencies covering [1, freq_max]
        log_fmax = jnp.log(jnp.array(self.freq_max, dtype=jnp.float32))
        k      = jnp.arange(half, dtype=jnp.float32)
        freqs  = jnp.exp(-log_fmax * k / jnp.maximum(half - 1, 1))   # [half]

        a_f    = jnp.array(a, dtype=jnp.float32)
        angles = a_f * freqs                                           # [half]
        sincos = jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=0)  # [2*half]

        # Two-layer projection: sincos → embed_dim
        h = jax.nn.silu(
            hk.Linear(self.embed_dim * 2, name="t_fc1")(sincos)
        )
        return hk.Linear(self.embed_dim, name="t_fc2")(h)             # [embed_dim]


# ==============================================================================
# 5. Haiku model
# ==============================================================================

class LagrangianForceCorrector(hk.Module):
    """
    Per-particle MLP that maps Lagrangian deformation features to a force
    correction (vector) or potential correction (scalar).

    Two time-conditioning modes
    ---------------------------
    use_film_conditioning=False  (default — backward compatible)
        a is broadcast to [N,1] and concatenated to the input.
        Only the FIRST layer sees the time signal directly.

    use_film_conditioning=True  (recommended)
        a → TimeEmbedding → emb_a [film_embed_dim]
        Each hidden layer l is modulated by emb_a via FiLM:

            h_l = (1 + γ_l) ⊙ fc_l(h_{l-1}) + β_l
            [γ_l, β_l] = film_proj_l(emb_a)     (zero-initialised)

        Zero-init ensures the model starts as a plain MLP (γ=0, β=0),
        and the time conditioning grows during training — no instability.

        Every hidden layer can learn a DIFFERENT response to cosmic time,
        enabling the model to capture epoch-specific phenomena (shell-crossing
        onset, virialization, multi-streaming) without a shared representation.

    Input per particle
    ------------------
    deform_feats : [N, D_feat]  strain / deformation features
    velocities   : [N, 3]       current velocities (context)
    scale_factors: scalar       scale factor a

    Output
    ------
    [N, output_dim]   output_dim=3 for direct force, 1 for scalar potential
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 3,
        output_dim: int = 3,
        use_film_conditioning: bool = False,
        film_embed_dim: int = 32,
        name: Optional[str] = None,
    ):
        super().__init__(name=name or "LagrangianForceCorrector")
        self.hidden_dim           = hidden_dim
        self.n_layers             = n_layers
        self.output_dim           = output_dim
        self.use_film_conditioning = use_film_conditioning
        self.film_embed_dim       = film_embed_dim

    # ------------------------------------------------------------------
    # Plain-concatenation path (original behaviour)
    # ------------------------------------------------------------------
    def _forward_concat(
        self,
        deform_feats: jnp.ndarray,
        velocities: jnp.ndarray,
        scale_factors,
    ) -> jnp.ndarray:
        N     = deform_feats.shape[0]
        a_col = jnp.ones((N, 1), dtype=jnp.float32) * scale_factors
        x     = jnp.concatenate([deform_feats, velocities, a_col], axis=-1)
        return hk.nets.MLP(
            output_sizes=[self.hidden_dim] * self.n_layers + [self.output_dim],
            activation=jax.nn.silu,
            name="lag_mlp",
        )(x)

    # ------------------------------------------------------------------
    # FiLM-conditioned path
    # ------------------------------------------------------------------
    def _forward_film(
        self,
        deform_feats: jnp.ndarray,
        velocities: jnp.ndarray,
        scale_factors,
    ) -> jnp.ndarray:
        # ── Time embedding ─────────────────────────────────────────────
        emb_a = TimeEmbedding(
            embed_dim=self.film_embed_dim, name="time_emb"
        )(scale_factors)                                    # [film_embed_dim]

        # ── Input projection (no a concatenated here) ──────────────────
        x = jnp.concatenate([deform_feats, velocities], axis=-1)   # [N, D_in]
        x = hk.Linear(self.hidden_dim, name="input_proj")(x)       # [N, D_h]

        # ── FiLM-conditioned hidden layers ─────────────────────────────
        # film_proj_{l} maps emb_a → [2·D_h] (γ_l ‖ β_l)
        # Zero-init: at step 0, γ_l=0 and β_l=0 → plain MLP behaviour.
        zero_w = hk.initializers.Constant(0.0)
        zero_b = hk.initializers.Constant(0.0)

        for l in range(self.n_layers):
            h = hk.Linear(self.hidden_dim, name=f"fc_{l}")(x)      # [N, D_h]

            # Per-layer FiLM projection (operates on the single emb_a vector)
            film = hk.Linear(
                2 * self.hidden_dim,
                w_init=zero_w, b_init=zero_b,
                name=f"film_{l}",
            )(emb_a)                                                 # [2·D_h]
            gamma, beta = jnp.split(film, 2, axis=-1)               # [D_h] each

            # Modulate: broadcast gamma/beta over N particles
            x = jax.nn.silu((1.0 + gamma) * h + beta)               # [N, D_h]

        # ── Output (linear — no activation for regression) ────────────
        return hk.Linear(self.output_dim, name="output")(x)         # [N, out]

    # ------------------------------------------------------------------
    def __call__(
        self,
        deform_feats: jnp.ndarray,
        velocities: jnp.ndarray,
        scale_factors,
    ) -> jnp.ndarray:
        if self.use_film_conditioning:
            return self._forward_film(deform_feats, velocities, scale_factors)
        return self._forward_concat(deform_feats, velocities, scale_factors)


# ==============================================================================
# 6. Factory
# ==============================================================================

def make_lagrangian_corrector(
    hidden_dim: int = 64,
    n_layers: int = 3,
    output_dim: int = 3,
    use_film_conditioning: bool = False,
    film_embed_dim: int = 32,
):
    """
    Returns hk.without_apply_rng(hk.transform(LagrangianCorr)).

    Parameters
    ----------
    hidden_dim            : width of each hidden layer
    n_layers              : number of hidden layers (total depth = n_layers + 1)
    output_dim            : 3 = direct force vector,  1 = scalar potential
    use_film_conditioning : if True, use FiLM time-conditioning (recommended)
    film_embed_dim        : dimension of the time embedding (ignored if False)

    Usage
    -----
        # Standalone (no FiLM)
        model = make_lagrangian_corrector(hidden_dim=64, n_layers=3, output_dim=3)

        # With FiLM time conditioning
        model = make_lagrangian_corrector(
            hidden_dim=64, n_layers=3, output_dim=3,
            use_film_conditioning=True, film_embed_dim=32,
        )

        params   = model.init(rng, deform_feats, velocities, scale_factor)
        delta_f  = model.apply(params, deform_feats, velocities, a)  # [N, 3]
    """
    def LagrangianCorr(deform_feats, velocities, scale_factors):
        return LagrangianForceCorrector(
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            output_dim=output_dim,
            use_film_conditioning=use_film_conditioning,
            film_embed_dim=film_embed_dim,
        )(deform_feats, velocities, scale_factors)

    return hk.without_apply_rng(hk.transform(LagrangianCorr))
