# ==============================================================================
# patched_transformer.py — Patched Lagrangian Transformer for PM force correction
#
# Architecture overview
# ---------------------
# 1.  WST enriches input grid  x → [128³, C + J*L]
# 2.  Each particle is embedded using its (pos, vel, a) + WST read at its position.
# 3.  Particles are sorted into 16³ = 4096 local patches (8³ mesh-cells each).
# 4.  A shared self-attention block (parameters fixed, vmapped over patches) reads
#     up to K=64 particles per patch and outputs enriched per-particle tokens.
# 5.  Per-particle transformer features are painted back to a grid via weighted
#     CIC, producing a feature field [128³, D_trans].
# 6.  The feature field is concatenated with the WST grid.
# 7.  CIC-read at the ORIGINAL (differentiable) particle positions extracts
#     per-particle features from the enriched grid.
# 8.  A small MLP maps those features → scalar ΔΦ per particle  [N, 1].
#
# Gradient path
# -------------
# Steps 2-5 operate on jax.lax.stop_gradient(positions) — the transformer
# context is fixed.  Steps 7-8 use the true (differentiable) positions through
# the CIC read.  As a result:
#
#     ∂ΔΦ_i / ∂pos_j = 0   for i ≠ j          (cross terms vanish)
#     ∂ΔΦ_i / ∂pos_i ≠ 0                      (sub-cell force via CIC grad)
#
# This lets pm.py compute per-particle forces with a single backward pass:
#
#     forces = jax.grad(lambda pos: jnp.sum(model(pos)[: ,0]))(positions)
#
# which equals [∂ΔΦ_i/∂pos_i for each i].
#
# Interface contract
# ------------------
#   PatchedTransformerCorr(x, positions, scale_factors, velocities)
#     x             : [128, 128, 128, C]   WST-enriched grid passed from pm.py
#     positions     : [N, 3]               mesh_lr units — do NOT mod inside here
#     scale_factors : scalar               scale factor a ∈ [0.1, 1.0]
#     velocities    : [N, 3]
#     returns       : [N, 1]               scalar ΔΦ per particle
# ==============================================================================

import jax
import jax.numpy as jnp
import haiku as hk
from typing import Optional

from jaxpm.painting import cic_paint, cic_read
from jaxpm.wst import WaveletScatteringTransform


class PatchedTransformerModel(hk.Module):
    """
    Patched Lagrangian Transformer.

    Hyper-parameters
    ----------------
    J, L          : WST scales and orientations  (total J*L Morlet channels)
    patch_size    : edge length of each local patch in mesh cells (default 8)
    K             : max particles gathered per patch (padded when < K)
    D_embed       : token embedding dimension
    D_hidden      : attention Q/K/V projection dimension
    D_trans       : number of per-particle features painted back to grid
    n_mlp_layers  : depth of the final ΔΦ MLP (minimum 1)
    grid_size     : edge length of the simulation mesh (128)
    """

    def __init__(
        self,
        J: int = 3,
        L: int = 4,
        patch_size: int = 8,
        K: int = 64,
        D_embed: int = 16,
        D_hidden: int = 32,
        D_trans: int = 8,
        n_mlp_layers: int = 2,
        grid_size: int = 128,
        name: Optional[str] = None,
    ):
        super().__init__(name=name or "PatchedTransformerModel")
        self.J = J
        self.L = L
        self.patch_size = patch_size
        self.K = K
        self.D_embed = D_embed
        self.D_hidden = D_hidden
        self.D_trans = D_trans
        self.n_mlp_layers = n_mlp_layers
        self.grid_size = grid_size

    def __call__(self, x, positions, scale_factors, velocities):
        G   = self.grid_size
        N   = positions.shape[0]
        PS  = self.patch_size
        K   = self.K
        D_e = self.D_embed
        D_h = self.D_hidden
        D_t = self.D_trans

        n_pd     = G // PS          # patches per dimension (e.g. 16)
        n_patches = n_pd ** 3       # total patches (e.g. 4096)

        # ── 1. WST enrichment ────────────────────────────────────────────────
        wst   = WaveletScatteringTransform(J=self.J, L=self.L, normalize=True, name="wst")
        x_wst = wst(x)              # [G, G, G, C + J*L]
        C_wst = x_wst.shape[-1]

        # ── 2. Stop-gradient block: transformer uses fixed positions ─────────
        # The only differentiable path to ΔΦ is the CIC-read in step 7.
        positions_sg  = jax.lax.stop_gradient(positions)
        velocities_sg = jax.lax.stop_gradient(velocities)

        # ── 3. Particle embedding ─────────────────────────────────────────────
        # Read WST features at each particle's (fixed) position
        # jax.vmap(cic_read, in_axes=(-1,None)) maps over channels → [C_wst, N]
        wst_at_particle = jax.vmap(cic_read, in_axes=(-1, None))(
            x_wst, positions_sg
        ).T                         # [N, C_wst]

        a_col           = jnp.ones((N, 1), dtype=jnp.float32) * scale_factors
        particle_input  = jnp.concatenate(
            [positions_sg, velocities_sg, a_col, wst_at_particle], axis=-1
        )                           # [N, 7 + C_wst]

        D_part_in = 7 + C_wst

        # Embedding matrix — registered before any vmap (Haiku-safe)
        W_embed = hk.get_parameter(
            "W_embed", [D_part_in, D_e],
            init=hk.initializers.VarianceScaling(1.0),
        )
        b_embed = hk.get_parameter("b_embed", [D_e], init=jnp.zeros)
        particle_tokens = jax.nn.relu(particle_input @ W_embed + b_embed)  # [N, D_e]

        # ── 4. Sort particles by patch ────────────────────────────────────────
        pos_mod   = jnp.mod(positions_sg, G)
        patch_3d  = jnp.floor(pos_mod / PS).astype(jnp.int32)           # [N, 3]
        patch_1d  = (
            patch_3d[:, 0] * n_pd * n_pd
            + patch_3d[:, 1] * n_pd
            + patch_3d[:, 2]
        )                                                                 # [N]

        sorted_order  = jnp.argsort(patch_1d, stable=True)               # [N]
        sorted_pos    = positions_sg[sorted_order]                        # [N, 3]
        sorted_tokens = particle_tokens[sorted_order]                     # [N, D_e]
        sorted_patch  = patch_1d[sorted_order]                           # [N]

        patch_counts = (
            jnp.zeros(n_patches, dtype=jnp.int32)
            .at[sorted_patch].add(jnp.ones(N, dtype=jnp.int32))
        )                                                                 # [n_patches]
        patch_starts = jnp.concatenate([
            jnp.array([0], dtype=jnp.int32),
            jnp.cumsum(patch_counts[:-1]),
        ])                                                                # [n_patches]

        # Patch geometric centres [n_patches, 3]
        p_3d         = jnp.stack(
            jnp.unravel_index(jnp.arange(n_patches), (n_pd, n_pd, n_pd)),
            axis=-1,
        )
        patch_centers = (p_3d.astype(jnp.float32) + 0.5) * PS           # [n_patches, 3]

        # ── 5. Shared attention parameters (defined before vmap) ─────────────
        # Relative-position encoder
        W_rel  = hk.get_parameter("W_rel",  [3, D_e],    init=hk.initializers.VarianceScaling(1.0))
        b_rel  = hk.get_parameter("b_rel",  [D_e],        init=jnp.zeros)

        # Single-layer self-attention
        W_q    = hk.get_parameter("W_q",    [D_e, D_h],  init=hk.initializers.VarianceScaling(1.0))
        W_k    = hk.get_parameter("W_k",    [D_e, D_h],  init=hk.initializers.VarianceScaling(1.0))
        W_v    = hk.get_parameter("W_v",    [D_e, D_h],  init=hk.initializers.VarianceScaling(1.0))
        W_attn = hk.get_parameter("W_attn", [D_h, D_e],  init=hk.initializers.VarianceScaling(1.0))
        b_attn = hk.get_parameter("b_attn", [D_e],        init=jnp.zeros)

        attn_scale = jnp.sqrt(jnp.array(D_h, dtype=jnp.float32))

        # ── 6. Per-patch self-attention (vmapped) ─────────────────────────────
        def process_patch(p):
            """Returns enriched tokens [K, D_e] for patch p."""
            start = patch_starts[p]
            count = jnp.minimum(patch_counts[p], K)

            # Clamp start so dynamic_slice never reads past the array end
            safe_start = jnp.minimum(start, jnp.maximum(N - K, 0))

            k_tokens = jax.lax.dynamic_slice_in_dim(sorted_tokens, safe_start, K, axis=0)  # [K, D_e]
            k_pos    = jax.lax.dynamic_slice_in_dim(sorted_pos,    safe_start, K, axis=0)  # [K, 3]

            # Boolean mask: True for real particles, False for padding
            mask = jnp.arange(K) < count                                                   # [K]

            # Sub-cell relative positions with periodic minimum-image convention
            rel_pos = k_pos - patch_centers[p]                                             # [K, 3]
            rel_pos = rel_pos - G * jnp.round(rel_pos / G)

            # Add relative-position encoding to token
            rel_enc = jax.nn.relu(rel_pos @ W_rel + b_rel)                                # [K, D_e]
            tokens  = k_tokens + rel_enc
            tokens  = jnp.where(mask[:, None], tokens, 0.0)

            # Self-attention
            Q = tokens @ W_q                                                               # [K, D_h]
            K_mat = tokens @ W_k                                                           # [K, D_h]
            V = tokens @ W_v                                                               # [K, D_h]

            scores    = (Q @ K_mat.T) / attn_scale                                        # [K, K]
            pair_mask = mask[:, None] & mask[None, :]
            scores    = jnp.where(pair_mask, scores, -1e9)
            attn_w    = jax.nn.softmax(scores, axis=-1)
            attn_w    = jnp.where(mask[:, None], attn_w, 0.0)

            h      = attn_w @ V                                                            # [K, D_h]
            out    = h @ W_attn + b_attn                                                   # [K, D_e]
            tokens = tokens + out                                                          # residual
            tokens = jnp.where(mask[:, None], tokens, 0.0)
            return tokens                                                                   # [K, D_e]

        # [n_patches, K, D_e]
        all_patch_tokens = jax.vmap(process_patch)(jnp.arange(n_patches, dtype=jnp.int32))

        # ── 7. Gather per-particle transformer features ───────────────────────
        # For sorted particle j: its patch is sorted_patch[j],
        # its local index within the patch is j - patch_starts[sorted_patch[j]].
        j_all         = jnp.arange(N)
        local_j       = j_all - patch_starts[sorted_patch]                # [N]
        local_j_clip  = jnp.clip(local_j, 0, K - 1)

        sorted_feats  = all_patch_tokens[sorted_patch, local_j_clip, :]   # [N, D_e]

        # Zero out particles that were truncated (local_j ≥ K, i.e. >K-th in their patch)
        sorted_feats  = jnp.where((local_j >= K)[:, None], 0.0, sorted_feats)

        # Un-sort back to original particle order
        inverse_sort  = (
            jnp.zeros(N, dtype=jnp.int32)
            .at[sorted_order].set(jnp.arange(N, dtype=jnp.int32))
        )                                                                  # [N]
        particle_feats = sorted_feats[inverse_sort]                        # [N, D_e]

        # ── 8. Project to D_trans and paint to feature grid ──────────────────
        W_proj = hk.get_parameter("W_proj", [D_e, D_t], init=hk.initializers.VarianceScaling(1.0))
        b_proj = hk.get_parameter("b_proj", [D_t],       init=jnp.zeros)
        trans_feats = jax.nn.relu(particle_feats @ W_proj + b_proj)       # [N, D_t]

        # Paint each feature channel to a [G, G, G] grid using weighted CIC.
        # vmap over channels (axis 1 of trans_feats), stack along last axis.
        feature_grid = jax.vmap(
            lambda fd: cic_paint(jnp.zeros((G, G, G)), positions_sg, weight=fd),
            in_axes=1,
            out_axes=-1,
        )(trans_feats)                                                     # [G, G, G, D_t]

        # ── 9. Enrich grid and CIC-read at differentiable positions ──────────
        x_enriched   = jnp.concatenate([x_wst, feature_grid], axis=-1)   # [G, G, G, C_wst+D_t]

        # This CIC-read uses the TRUE (differentiable) positions — this is
        # where the sub-cell gradient path lives.
        feat_at_pos  = jax.vmap(cic_read, in_axes=(-1, None))(
            x_enriched, positions
        ).T                                                                # [N, C_wst+D_t]

        # ── 10. MLP: features + vel + a → scalar ΔΦ per particle ─────────────
        # velocities are fixed context (not in gradient path)
        mlp_input = jnp.concatenate([
            feat_at_pos,
            velocities_sg,
            jnp.ones((N, 1), dtype=jnp.float32) * scale_factors,
        ], axis=-1)                                                        # [N, C_wst+D_t+4]

        D_mlp_in = C_wst + D_t + 4
        hidden_sizes = [D_e] * self.n_mlp_layers + [1]

        delta_phi = hk.nets.MLP(
            output_sizes=hidden_sizes,
            activation=jax.nn.relu,
            name="mlp_phi",
        )(mlp_input)                                                       # [N, 1]

        return delta_phi


# ==============================================================================
# Public entry-point: matches the CNN+WST call signature used in build_network
# ==============================================================================

def make_patched_transformer(
    J: int = 3,
    L: int = 4,
    patch_size: int = 8,
    K: int = 64,
    D_embed: int = 16,
    D_hidden: int = 32,
    D_trans: int = 8,
    n_mlp_layers: int = 2,
    grid_size: int = 128,
):
    """
    Returns a transformed Haiku model (hk.without_apply_rng(hk.transform(...))).

    Usage in build_network():
        model = make_patched_transformer(**config_kwargs)
        params = model.init(rng, x, positions, scale_factor, velocities)
        delta_phi = model.apply(params, x, positions, a, velocities)  # [N,1]
    """

    def PatchedTransformerCorr(x, positions, scale_factors, velocities):
        return PatchedTransformerModel(
            J=J,
            L=L,
            patch_size=patch_size,
            K=K,
            D_embed=D_embed,
            D_hidden=D_hidden,
            D_trans=D_trans,
            n_mlp_layers=n_mlp_layers,
            grid_size=grid_size,
        )(x, positions, scale_factors, velocities)

    return hk.without_apply_rng(hk.transform(PatchedTransformerCorr))


# ==============================================================================
# HybridTransformerModel — Eulerian + Lagrangian dual-branch correction
# ==============================================================================
#
# Physical motivation
# -------------------
# The CNN-WST branch already captures *Eulerian* corrections well: it reads the
# local density + tidal environment from the grid and outputs a smooth potential
# correction.  What it cannot resolve is *multi-streaming*: in virialized
# regions, particles from different Lagrangian origins occupy the same Eulerian
# volume with disparate velocities.  The density field is degenerate over these
# multi-stream configurations, so any purely Eulerian model is blind to them.
#
# The Lagrangian transformer branch operates exclusively on particle-level data
# (position, velocity, scale factor) — no grid features.  Its key innovation is
# a **velocity-aware relative encoding**:
#
#     rel_feat_ij = [Δpos_ij / (PS/2),  Δvel_ij / σ_v]  ∈ ℝ⁶
#
# where σ_v is the global RMS velocity and Δvel_ij = vel_i − ⟨vel⟩_patch.
# Attention weights thus encode both spatial proximity AND kinematic coherence:
#   • high weight  →  same infalling stream (small Δpos, small Δvel)
#   • low weight   →  different streams or virialized scatter (large Δvel)
#
# Gradient path (identical to PatchedTransformerModel)
# ----------------------------------------------------
# Both branches end with a CIC read at the TRUE (differentiable) positions,
# so jax.grad(Σ ΔΦ)(positions) gives [∂ΔΦ_i/∂pos_i] in one backward pass.
# The transformer context uses stop_gradient(positions), making cross-derivatives
# zero.
#
# Combination modes
# -----------------
#   "sum"    ΔΦ = MLP_E(f_E, v, a) + MLP_L(f_L, v, a)
#              Two independent potential corrections; interpretable and easy to
#              initialise near zero for the Lagrangian branch.
#   "concat" ΔΦ = MLP([f_E, f_L, v, a])
#              Single expressive MLP; allows cross-branch interactions.
#
# Interface
# ---------
#   HybridTransformerModel(x, positions, scale_factors, velocities)
#     x             : [G, G, G, C]   grid passed from pm.py  (pm_pot + delta)
#     positions     : [N, 3]         mesh_lr units
#     scale_factors : scalar
#     velocities    : [N, 3]
#     returns       : [N, 1]         scalar ΔΦ per particle
# ==============================================================================


class HybridTransformerModel(hk.Module):
    """Dual-branch Eulerian + velocity-aware Lagrangian transformer."""

    def __init__(
        self,
        J: int = 3,
        L: int = 4,
        patch_size: int = 8,
        K: int = 64,
        D_embed: int = 16,
        D_hidden: int = 32,
        D_trans: int = 8,
        n_mlp_layers: int = 2,
        grid_size: int = 128,
        combine_mode: str = "sum",   # "sum" | "concat"
        name: Optional[str] = None,
    ):
        super().__init__(name=name or "HybridTransformerModel")
        self.J             = J
        self.L             = L
        self.patch_size    = patch_size
        self.K             = K
        self.D_embed       = D_embed
        self.D_hidden      = D_hidden
        self.D_trans       = D_trans
        self.n_mlp_layers  = n_mlp_layers
        self.grid_size     = grid_size
        self.combine_mode  = combine_mode

    def __call__(self, x, positions, scale_factors, velocities):
        G   = self.grid_size
        N   = positions.shape[0]
        PS  = self.patch_size
        K   = self.K
        D_e = self.D_embed
        D_h = self.D_hidden
        D_t = self.D_trans

        n_pd      = G // PS
        n_patches = n_pd ** 3

        # ── Stop-gradient block ───────────────────────────────────────────────
        positions_sg  = jax.lax.stop_gradient(positions)
        velocities_sg = jax.lax.stop_gradient(velocities)

        # ════════════════════════════════════════════════════════════════════
        # Branch 1 — Eulerian (WST grid read at differentiable positions)
        # ════════════════════════════════════════════════════════════════════
        wst   = WaveletScatteringTransform(J=self.J, L=self.L, normalize=True, name="wst")
        x_wst = wst(x)          # [G, G, G, C_wst]
        C_wst = x_wst.shape[-1]

        # CIC read at TRUE positions — this IS the Eulerian gradient path
        feat_E = jax.vmap(cic_read, in_axes=(-1, None))(x_wst, positions).T  # [N, C_wst]

        # ════════════════════════════════════════════════════════════════════
        # Branch 2 — Lagrangian transformer (pure particle data, no grid)
        # ════════════════════════════════════════════════════════════════════

        # ── 2a. Global velocity scale (for relative-velocity normalisation) ─
        vel_scale = jnp.sqrt(jnp.mean(velocities_sg ** 2) + 1e-8)  # scalar

        # ── 2b. Particle tokens: pos + vel + a (7-dim, no grid features) ───
        a_col          = jnp.ones((N, 1), dtype=jnp.float32) * scale_factors
        particle_input = jnp.concatenate(
            [positions_sg / G, velocities_sg / (vel_scale + 1e-8), a_col], axis=-1
        )   # [N, 7]  — normalised so all features are O(1)

        W_embed = hk.get_parameter(
            "W_embed", [7, D_e], init=hk.initializers.VarianceScaling(1.0)
        )
        b_embed = hk.get_parameter("b_embed", [D_e], init=jnp.zeros)
        particle_tokens = jax.nn.relu(particle_input @ W_embed + b_embed)   # [N, D_e]

        # ── 2c. Sort particles into Eulerian patches ─────────────────────────
        pos_mod   = jnp.mod(positions_sg, G)
        patch_3d  = jnp.floor(pos_mod / PS).astype(jnp.int32)
        patch_1d  = (
            patch_3d[:, 0] * n_pd * n_pd
            + patch_3d[:, 1] * n_pd
            + patch_3d[:, 2]
        )

        sorted_order  = jnp.argsort(patch_1d, stable=True)
        sorted_pos    = positions_sg[sorted_order]
        sorted_vel    = velocities_sg[sorted_order]
        sorted_tokens = particle_tokens[sorted_order]
        sorted_patch  = patch_1d[sorted_order]

        patch_counts = (
            jnp.zeros(n_patches, dtype=jnp.int32)
            .at[sorted_patch].add(jnp.ones(N, dtype=jnp.int32))
        )
        patch_starts = jnp.concatenate([
            jnp.array([0], dtype=jnp.int32),
            jnp.cumsum(patch_counts[:-1]),
        ])

        p_3d         = jnp.stack(
            jnp.unravel_index(jnp.arange(n_patches), (n_pd, n_pd, n_pd)), axis=-1
        )
        patch_centers = (p_3d.astype(jnp.float32) + 0.5) * PS   # [n_patches, 3]

        # ── 2d. Per-patch mean velocity (computed before vmap) ───────────────
        # Used as the velocity reference frame for each patch.
        # vel_sum[p] = Σ_{i in patch p} vel_i  (sorted order)
        vel_sum = (
            jnp.zeros((n_patches, 3), dtype=jnp.float32)
            .at[sorted_patch].add(sorted_vel)
        )
        patch_vel_mean = vel_sum / jnp.maximum(
            patch_counts[:, None].astype(jnp.float32), 1.0
        )                                                          # [n_patches, 3]

        # ── 2e. Shared attention parameters (defined before vmap) ────────────
        # Velocity-aware relative encoding: 6 = 3 (pos) + 3 (vel)
        W_rel  = hk.get_parameter("W_rel",  [6, D_e],   init=hk.initializers.VarianceScaling(1.0))
        b_rel  = hk.get_parameter("b_rel",  [D_e],       init=jnp.zeros)

        W_q    = hk.get_parameter("W_q",    [D_e, D_h], init=hk.initializers.VarianceScaling(1.0))
        W_k    = hk.get_parameter("W_k",    [D_e, D_h], init=hk.initializers.VarianceScaling(1.0))
        W_v    = hk.get_parameter("W_v",    [D_e, D_h], init=hk.initializers.VarianceScaling(1.0))
        W_attn = hk.get_parameter("W_attn", [D_h, D_e], init=hk.initializers.VarianceScaling(1.0))
        b_attn = hk.get_parameter("b_attn", [D_e],       init=jnp.zeros)

        attn_scale = jnp.sqrt(jnp.array(D_h, dtype=jnp.float32))

        # ── 2f. Per-patch self-attention (vmapped) ───────────────────────────
        def process_patch(p):
            start      = patch_starts[p]
            count      = jnp.minimum(patch_counts[p], K)
            safe_start = jnp.minimum(start, jnp.maximum(N - K, 0))

            k_tokens = jax.lax.dynamic_slice_in_dim(sorted_tokens, safe_start, K, axis=0)
            k_pos    = jax.lax.dynamic_slice_in_dim(sorted_pos,    safe_start, K, axis=0)
            k_vel    = jax.lax.dynamic_slice_in_dim(sorted_vel,    safe_start, K, axis=0)

            mask = jnp.arange(K) < count   # [K]  True = real particle

            # Velocity-aware relative encoding
            # rel_pos: displacement from patch centre (periodic)
            rel_pos = k_pos - patch_centers[p]
            rel_pos = rel_pos - G * jnp.round(rel_pos / G)

            # rel_vel: deviation from patch mean velocity
            # Small rel_vel → same dynamical stream; large → multi-streaming
            rel_vel = k_vel - patch_vel_mean[p]

            # Normalise both to O(1) and concatenate → [K, 6]
            rel_feat = jnp.concatenate([
                rel_pos / (PS / 2.0 + 1e-6),
                rel_vel / (vel_scale + 1e-8),
            ], axis=-1)

            rel_enc = jax.nn.relu(rel_feat @ W_rel + b_rel)   # [K, D_e]
            tokens  = k_tokens + rel_enc
            tokens  = jnp.where(mask[:, None], tokens, 0.0)

            # Single-head self-attention
            Q      = tokens @ W_q
            K_mat  = tokens @ W_k
            V      = tokens @ W_v

            scores    = (Q @ K_mat.T) / attn_scale
            pair_mask = mask[:, None] & mask[None, :]
            scores    = jnp.where(pair_mask, scores, -1e9)
            attn_w    = jax.nn.softmax(scores, axis=-1)
            attn_w    = jnp.where(mask[:, None], attn_w, 0.0)

            h      = attn_w @ V
            out    = h @ W_attn + b_attn
            tokens = tokens + out           # residual connection
            tokens = jnp.where(mask[:, None], tokens, 0.0)
            return tokens                   # [K, D_e]

        all_patch_tokens = jax.vmap(process_patch)(
            jnp.arange(n_patches, dtype=jnp.int32)
        )   # [n_patches, K, D_e]

        # ── 2g. Gather per-particle transformer features ─────────────────────
        j_all        = jnp.arange(N)
        local_j      = j_all - patch_starts[sorted_patch]
        local_j_clip = jnp.clip(local_j, 0, K - 1)

        sorted_feats = all_patch_tokens[sorted_patch, local_j_clip, :]
        sorted_feats = jnp.where((local_j >= K)[:, None], 0.0, sorted_feats)

        inverse_sort  = (
            jnp.zeros(N, dtype=jnp.int32)
            .at[sorted_order].set(jnp.arange(N, dtype=jnp.int32))
        )
        particle_feats = sorted_feats[inverse_sort]   # [N, D_e]

        # ── 2h. Project + paint to Lagrangian feature grid ──────────────────
        W_proj = hk.get_parameter(
            "W_proj", [D_e, D_t], init=hk.initializers.VarianceScaling(1.0)
        )
        b_proj = hk.get_parameter("b_proj", [D_t], init=jnp.zeros)
        trans_feats = jax.nn.relu(particle_feats @ W_proj + b_proj)   # [N, D_t]

        feature_grid_L = jax.vmap(
            lambda fd: cic_paint(jnp.zeros((G, G, G)), positions_sg, weight=fd),
            in_axes=1, out_axes=-1,
        )(trans_feats)   # [G, G, G, D_t]

        # CIC read at TRUE positions — Lagrangian gradient path
        feat_L = jax.vmap(cic_read, in_axes=(-1, None))(
            feature_grid_L, positions
        ).T   # [N, D_t]

        # ════════════════════════════════════════════════════════════════════
        # Combine branches → scalar ΔΦ per particle
        # ════════════════════════════════════════════════════════════════════
        vel_sg_mlp = velocities_sg
        a_mlp      = jnp.ones((N, 1), dtype=jnp.float32) * scale_factors
        hidden_sz  = [D_e] * self.n_mlp_layers + [1]

        if self.combine_mode == "sum":
            # Two independent scalar potentials — each branch fully autonomous.
            # Initialise mlp_L near zero (small VarianceScaling) so the model
            # starts as a pure Eulerian correction and learns Lagrangian terms.
            mlp_E_in   = jnp.concatenate([feat_E, vel_sg_mlp, a_mlp], axis=-1)
            delta_phi_E = hk.nets.MLP(
                hidden_sz, activation=jax.nn.relu, name="mlp_E"
            )(mlp_E_in)   # [N, 1]

            mlp_L_in   = jnp.concatenate([feat_L, vel_sg_mlp, a_mlp], axis=-1)
            delta_phi_L = hk.nets.MLP(
                hidden_sz, activation=jax.nn.relu, name="mlp_L"
            )(mlp_L_in)   # [N, 1]

            return delta_phi_E + delta_phi_L

        else:  # "concat" — richer cross-branch interactions
            mlp_in = jnp.concatenate([feat_E, feat_L, vel_sg_mlp, a_mlp], axis=-1)
            return hk.nets.MLP(
                hidden_sz, activation=jax.nn.relu, name="mlp_phi"
            )(mlp_in)   # [N, 1]


# ==============================================================================
# Public factory: make_hybrid_transformer
# ==============================================================================

def make_hybrid_transformer(
    J: int = 3,
    L: int = 4,
    patch_size: int = 8,
    K: int = 64,
    D_embed: int = 16,
    D_hidden: int = 32,
    D_trans: int = 8,
    n_mlp_layers: int = 2,
    grid_size: int = 128,
    combine_mode: str = "sum",
):
    """
    Returns hk.without_apply_rng(hk.transform(HybridTransformerCorr)).

    Usage:
        model  = make_hybrid_transformer(**config_kwargs)
        params = model.init(rng, x, positions, scale_factor, velocities)
        dphi   = model.apply(params, x, positions, a, velocities)  # [N,1]

    combine_mode:
        "sum"    — ΔΦ_E + ΔΦ_L  (interpretable, Lagrangian branch starts near 0)
        "concat" — single MLP over [f_E, f_L, vel, a]  (more expressive)
    """

    def HybridTransformerCorr(x, positions, scale_factors, velocities):
        return HybridTransformerModel(
            J=J, L=L,
            patch_size=patch_size,
            K=K,
            D_embed=D_embed,
            D_hidden=D_hidden,
            D_trans=D_trans,
            n_mlp_layers=n_mlp_layers,
            grid_size=grid_size,
            combine_mode=combine_mode,
        )(x, positions, scale_factors, velocities)

    return hk.without_apply_rng(hk.transform(HybridTransformerCorr))