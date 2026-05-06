import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import jax.numpy as jnp
from jaxpm.pm import get_delta
import numpy as np
import jax

from scipy.stats import pearsonr

def plot_density_comparison(pos_hr_t, pos_lr_t, mesh_plot=512, cmap="cividis"):
    mesh_plot = 512
    cmap = "cividis"

    delta_hr = get_delta(pos_hr_t * 4, (mesh_plot, mesh_plot, mesh_plot))
    delta_lr = get_delta(pos_lr_t * 4, (mesh_plot, mesh_plot, mesh_plot))

    # 1. Proyectamos primero para no calcular mil veces
    proj_hr = delta_hr.sum(axis=0)
    proj_lr = delta_lr.sum(axis=0)

    # 2. Evitamos valores <= 0 para el Log (muy importante)
    # Le sumamos un valor pequeño o truncamos
    proj_hr = jnp.where(proj_hr <= 0, 1e-6, proj_hr)
    proj_lr = jnp.where(proj_lr <= 0, 1e-6, proj_lr)

    # 3. Calculamos la norma usando percentiles para "abrir" el contraste
    # Usamos el 1% para el mínimo y el 99% para el máximo
    vmin_val = float(jnp.percentile(proj_hr, 1))
    vmax_val = float(jnp.percentile(proj_hr, 99))

    norm = LogNorm(
        vmin=max(1e-3, vmin_val),  # Aseguramos que no sea 0
        vmax=vmax_val,
    )

    delta_lr = get_delta(pos_lr_t * 4, (mesh_plot, mesh_plot, mesh_plot))

    fig, ax = plt.subplots(ncols=2, figsize=(10, 5), facecolor="#0D1117")
    ax[0].imshow((delta_lr[:, :, :]).sum(axis=0), norm=norm, cmap=cmap)

    ax[1].imshow((delta_hr[:, :, :]).sum(axis=0), norm=norm, cmap=cmap)

    ax[0].set_title("LR")
    ax[1].set_title("HR")

    return fig, ax


# ───────────────────────────────────────────────────────────────────────────
# Cell 1 — 3-D scatter: particles in sub-volume coloured by Lagrangian
# neighbour direction.
#
# What to look for
# ────────────────
# • At early times (small a) neighbours should be close to the central
#   particle — the lattice is barely displaced.
# • At late times, neighbours may have crossed each other (shell-crossing),
#   which means the *Lagrangian* ±x neighbour may now sit far away or even
#   on the OPPOSITE side of the particle in Eulerian space.
# • nbrs_outside counts how many Lagrangian neighbours are outside the
#   sub-volume — a proxy for how much the lattice has been disrupted.
# ───────────────────────────────────────────────────────────────────────────


def plot_lagrangian_neighbors(
    pos_jax,
    neighbor_idx,
    mesh_size,
    a_val,
    sv_center=(64.0, 64.0, 64.0),
    sv_half=12.0,
    n_central=30,
    seed=42,
):
    """
    Visualiza las partículas centrales y sus 6 vecinos lagrangianos dentro de un sub-volumen.
    """
    # ── Configuración Estética ────────────────────────────────────────────────
    DIR_LABELS = ["+x", "−x", "+y", "−y", "+z", "−z"]
    DIR_COLORS = ["#E63946", "#FF8C00", "#2EC4B6", "#118AB2", "#8338EC", "#FB5607"]

    SV_CENTER = np.array(sv_center)

    # ── Preparación de Datos ──────────────────────────────────────────────────
    # Convertir a numpy y aplicar condiciones periódicas
    pos_np = np.asarray(jax.device_get(pos_jax)) % mesh_size
    nbr_np = np.asarray(neighbor_idx)

    # Máscara para partículas dentro del sub-volumen
    in_sv = np.all(np.abs(pos_np - SV_CENTER) < sv_half, axis=1)
    sv_idx = np.where(in_sv)[0]

    if len(sv_idx) == 0:
        print("⚠️ No hay partículas en el sub-volumen seleccionado.")
        return

    # Selección aleatoria de partículas centrales
    rng_vis = np.random.default_rng(seed)
    sel_idx = rng_vis.choice(sv_idx, size=min(n_central, len(sv_idx)), replace=False)

    # ── Creación de la Figura ─────────────────────────────────────────────────
    fig = plt.figure(figsize=(13, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#0D1117")
    fig.patch.set_facecolor("#0D1117")

    # 1. Partículas de contexto (todas las del sub-volumen)
    ax.scatter(*pos_np[sv_idx].T, s=0.8, c="#444455", alpha=0.25, rasterized=True)

    # 2. Vecinos de partículas centrales
    for k, (label, color) in enumerate(zip(DIR_LABELS, DIR_COLORS)):
        nbr_global = nbr_np[sel_idx, k]
        nbr_pos = pos_np[nbr_global]
        ax.scatter(
            *nbr_pos.T,
            s=28,
            c=color,
            alpha=0.85,
            label=f"nbr {label}",
            edgecolors="none",
            zorder=4,
        )

    # 3. Líneas de conexión (con manejo de imagen mínima periódica)
    for ci in sel_idx:
        cpos = pos_np[ci]
        for k, color in enumerate(DIR_COLORS):
            npos = pos_np[nbr_np[ci, k]]
            d = npos - cpos
            d -= mesh_size * np.round(d / mesh_size)  # Wrap periódico
            ax.plot(
                [cpos[0], cpos[0] + d[0]],
                [cpos[1], cpos[1] + d[1]],
                [cpos[2], cpos[2] + d[2]],
                c=color,
                lw=0.6,
                alpha=0.20,
            )

    # 4. Partículas centrales (en blanco)
    ax.scatter(
        *pos_np[sel_idx].T,
        s=55,
        c="white",
        edgecolors="#CCCCCC",
        linewidths=0.4,
        zorder=6,
        label="central",
    )

    # 5. Dibujar la caja del sub-volumen
    lo, hi = SV_CENTER - sv_half, SV_CENTER + sv_half
    for xs in [(lo[0], hi[0])]:
        for ys in [(lo[1], hi[1])]:
            ax.plot([xs[0], xs[0]], [ys[0], ys[0]], [lo[2], hi[2]], c="#555577", lw=0.5)
            ax.plot([xs[0], xs[0]], [ys[1], ys[1]], [lo[2], hi[2]], c="#555577", lw=0.5)
    for zs in [(lo[2], hi[2])]:
        ax.plot([lo[0], hi[0]], [lo[1], lo[1]], [zs[0], zs[0]], c="#555577", lw=0.5)
        ax.plot([lo[0], hi[0]], [hi[1], hi[1]], [zs[0], zs[0]], c="#555577", lw=0.5)

    # ── Formateo de Ejes ──────────────────────────────────────────────────────
    ax.set_xlabel("x [mesh cells]", color="white", labelpad=6)
    ax.set_ylabel("y [mesh cells]", color="white", labelpad=6)
    ax.set_zlabel("z [mesh cells]", color="white", labelpad=6)
    ax.tick_params(colors="white", labelsize=7)

    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("#222233")
    ax.grid(False)

    # Leyenda
    leg = ax.legend(
        loc="upper left",
        fontsize=8,
        framealpha=0.15,
        labelcolor="white",
        markerscale=1.6,
    )
    leg.get_frame().set_facecolor("#1A1A2E")

    ax.set_title(
        f"Vecinos Lagrangianos | a = {a_val:.3f} | "
        f"{len(sel_idx)} partículas centrales · {len(sv_idx):,} en sub-volumen",
        color="white",
        fontsize=11,
        pad=12,
    )

    plt.tight_layout()
    plt.show()

    # Diagnóstico
    nbrs_outside = sum(not in_sv[nbr_np[ci, k]] for ci in sel_idx for k in range(6))
    total_nbrs = len(sel_idx) * 6
    print(
        f"Vecinos fuera del sub-volumen euleriano: {nbrs_outside} / {total_nbrs} "
        f"({nbrs_outside / total_nbrs:.1%})"
    )


# Ejemplo de uso:
# plot_lagrangian_neighbors(pos_lr_t, neighbor_idx, mesh_lr, a_train)


def plot_deformation_features(
    pos_jax,
    strain_mag,
    det_D,
    force_mag,
    mesh_size,
    a_val,
    sv_center=(64.0, 64.0, 64.0),
    sv_half=12.0,
):
    """
    Genera 3 paneles proyectados en 2D para diagnosticar la correlación entre
    la deformación (strain) y el residual de fuerza.
    """
    # ── Preparación de Datos ──────────────────────────────────────────────────
    # Aseguramos que los datos estén en CPU y formato numpy
    pos_np = np.asarray(jax.device_get(pos_jax)) % mesh_size
    sm = np.asarray(strain_mag)
    dD = np.asarray(det_D)
    fm = np.asarray(force_mag)

    cx, cy, cz = sv_center

    # Máscara del sub-volumen
    sv_mask = (
        (np.abs(pos_np[:, 0] - cx) < sv_half)
        & (np.abs(pos_np[:, 1] - cy) < sv_half)
        & (np.abs(pos_np[:, 2] - cz) < sv_half)
    )

    idx = np.where(sv_mask)[0]
    if len(idx) == 0:
        print("⚠️ Sub-volumen vacío. Ajusta sv_center o sv_half.")
        return

    # Extraer datos del sub-volumen
    px = pos_np[idx, 0]
    py = pos_np[idx, 1]
    sm_sub = sm[idx]
    dD_sub = dD[idx]
    fm_sub = fm[idx]

    # ── Cálculos Estadísticos (Pearson R) ─────────────────────────────────────
    r_sf, p_sf = pearsonr(sm_sub, fm_sub)
    r_df, p_df = pearsonr(np.abs(dD_sub), fm_sub)
    # Correlación con el flag de Shell-Crossing (SC)
    sc_flag = (dD_sub < 0).astype(float)
    r_sc, p_sc = pearsonr(sc_flag, fm_sub)

    # ── Graficado ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor="white")

    # Panel 1: Magnitud de Strain ||E||_F
    vmax_sm = np.percentile(sm_sub, 98)
    sc1 = axes[0].scatter(
        px, py, c=sm_sub, s=8, cmap="inferno", alpha=0.85, vmin=0, vmax=vmax_sm
    )
    plt.colorbar(sc1, ax=axes[0], label="||E||_F")
    axes[0].set_title(f"Strain magnitude ||E||_F\nR(||E||_F, |ΔF|) = {r_sf:.3f}")
    axes[0].set_xlabel("x [mesh units]")
    axes[0].set_ylabel("y [mesh units]")
    axes[0].set_aspect("equal")

    # Panel 2: det(D) — Jacobiano (Divergente en 0)
    clim = max(abs(np.percentile(dD_sub, 1)), abs(np.percentile(dD_sub, 99)))
    sc2 = axes[1].scatter(
        px, py, c=dD_sub, s=8, cmap="RdBu", alpha=0.85, vmin=-clim, vmax=clim
    )
    plt.colorbar(sc2, ax=axes[1], label="det(D)")

    # Marcar Shell-Crossing (det < 0)
    sc_mask = dD_sub < 0
    axes[1].scatter(
        px[sc_mask],
        py[sc_mask],
        s=12,
        c="lime",
        marker="x",
        linewidths=0.6,
        alpha=0.7,
        label=f"SC ({sc_mask.sum():,})",
    )
    axes[1].legend(fontsize=8, loc="upper right")
    axes[1].set_title(
        f"det(D) | SC frac={sc_mask.mean():.2%}\nR(|det D|, |ΔF|) = {r_df:.3f}"
    )
    axes[1].set_xlabel("x [mesh units]")
    axes[1].set_aspect("equal")

    # Panel 3: |ΔF| — El objetivo a predecir
    vmax_fm = np.percentile(fm_sub, 98)
    sc3 = axes[2].scatter(
        px, py, c=fm_sub, s=8, cmap="viridis", alpha=0.85, vmin=0, vmax=vmax_fm
    )
    plt.colorbar(sc3, ax=axes[2], label="|ΔF|")
    axes[2].set_title(f"|ΔF| (Target Residual)\nR(SC flag, |ΔF|) = {r_sc:.3f}")
    axes[2].set_xlabel("x [mesh units]")
    axes[2].set_aspect("equal")

    fig.suptitle(
        f"Deformation Features | Sub-volume ±{sv_half} | a={a_val:.3f} | n={len(idx):,} particles",
        fontsize=13,
        y=1.02,
    )

    plt.tight_layout()
    plt.show()

    # ── Diagnóstico en Consola ───────────────────────────────────────────────
    print("\n── Signal diagnostic ──────────────────────────────────────")
    print(f"  R( ||E||_F , |ΔF| ) = {r_sf:+.4f}   p={p_sf:.2e}")
    print(f"  R( |det D| , |ΔF| ) = {r_df:+.4f}   p={p_df:.2e}")
    print(f"  R( SC flag , |ΔF| ) = {r_sc:+.4f}   p={p_sc:.2e}")

    verdict = (
        "SIGNAL PRESENT ✓"
        if abs(r_sf) > 0.15
        else "WEAK SIGNAL — consider more features"
    )
    print(f"  → {verdict}\n")


# Ejemplo de uso:
# plot_deformation_features(pos_lr_t, strain_mag_train, det_D_train, force_mag_train, mesh_lr, a_train)

def plot_deformation_features_hex(pos_jax, strain_mag, det_D, force_mag, mesh_size, a_val,
                                  sv_center=(64.0, 64.0, 64.0), 
                                  sv_half=12.0, 
                                  gridsize=60):
    """
    Diagnóstico de deformación usando HEXBIN para suavizar el ruido y 
    manejar grandes densidades de partículas.
    """
    # ── Preparación de Datos ──────────────────────────────────────────────────
    pos_np = np.asarray(jax.device_get(pos_jax)) % mesh_size
    sm     = np.asarray(strain_mag)
    dD     = np.asarray(det_D)
    fm     = np.asarray(force_mag)
    
    cx, cy, cz = sv_center

    # Máscara del sub-volumen
    sv_mask = (
        (np.abs(pos_np[:, 0] - cx) < sv_half) &
        (np.abs(pos_np[:, 1] - cy) < sv_half) &
        (np.abs(pos_np[:, 2] - cz) < sv_half)
    )
    
    idx = np.where(sv_mask)[0]
    if len(idx) == 0:
        print("⚠️ Sub-volumen vacío.")
        return

    px, py = pos_np[idx, 0], pos_np[idx, 1]
    sm_sub, dD_sub, fm_sub = sm[idx], dD[idx], fm[idx]

    # Cálculos estadísticos
    r_sf, _ = pearsonr(sm_sub, fm_sub)
    r_df, _ = pearsonr(np.abs(dD_sub), fm_sub)
    sc_mask = dD_sub < 0

    # ── Graficado con Hexbin ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor="white")
    
    # Común para todos los hexbins: mincnt=1 evita pintar hexágonos vacíos
    hb_kwargs = dict(gridsize=gridsize, mincnt=1, edgecolors='none', extent=[cx-sv_half, cx+sv_half, cy-sv_half, cy+sv_half])

    # Panel 1: ||E||_F
    vmax_sm = np.percentile(sm_sub, 98)
    hb1 = axes[0].hexbin(px, py, C=sm_sub, reduce_C_function=np.mean, cmap="inferno", vmin=0, vmax=vmax_sm, **hb_kwargs)
    axes[0].set_title(f"Strain ||E||_F (Avg)\nR={r_sf:.3f}")
    plt.colorbar(hb1, ax=axes[0], label="||E||_F")

    # Panel 2: det(D) - Jacobiano
    clim = max(abs(np.percentile(dD_sub, 1)), abs(np.percentile(dD_sub, 99)))
    hb2 = axes[1].hexbin(px, py, C=dD_sub, reduce_C_function=np.mean, cmap="RdBu", vmin=-clim, vmax=clim, **hb_kwargs)
    axes[1].set_title(f"det(D) (Avg)\nShell-cross: {sc_mask.mean():.2%}")
    plt.colorbar(hb2, ax=axes[1], label="det(D)")
    
    # Overlay de Shell-Crossing (opcional, puntos donde ocurre SC)
    if sc_mask.any():
        axes[1].scatter(px[sc_mask], py[sc_mask], s=2, c="lime", alpha=0.3, label="SC")

    # Panel 3: |ΔF| - Target
    vmax_fm = np.percentile(fm_sub, 98)
    hb3 = axes[2].hexbin(px, py, C=fm_sub, reduce_C_function=np.mean, cmap="viridis", vmin=0, vmax=vmax_fm, **hb_kwargs)
    axes[2].set_title(f"Target |ΔF| (Avg)\nR(|det D|, |ΔF|) = {r_df:.3f}")
    plt.colorbar(hb3, ax=axes[2], label="|ΔF|")

    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlabel("x [mesh units]")
        ax.set_facecolor("#f0f0f0") # Color de fondo para ver el grid de hexágonos

    axes[0].set_ylabel("y [mesh units]")
    fig.suptitle(f"Hexbin Analysis | a={a_val:.3f} | n={len(idx):,} particles", fontsize=14, y=1.05)
    
    plt.tight_layout()
    plt.show()

# ───────────────────────────────────────────────────────────────────────────
# Cell 2 — 2-D deformation feature panels
#
# Three projected panels in the x-y plane of the sub-volume:
#   Left:   ||E||_F  — total strain magnitude (dark=compressed, bright=extended)
#   Centre: det(D)   — Jacobian; red < 0 (shell-crossing), blue > 0 (normal)
#   Right:  |ΔF|     — magnitude of force residual we want to predict
#
# Pearson R between ||E||_F and |ΔF| is the key diagnostic:
#   R > 0.3 → deformation tensor contains signal worth learning from.
# ───────────────────────────────────────────────────────────────────────────

def plot_lagrangian_features_analysis(pos_jax, feats_jax, det_D_jax, delta_f_jax, mesh_size, a_val,
                                      sv_center=(64.0, 64.0, 64.0), 
                                      sv_half=12.0, 
                                      n_sel=30):
    """
    Analiza y visualiza las features de deformación y su correlación con el target de fuerza
    en un sub-volumen específico.
    """
    # ── 1. Preparación y Derivación de Cantidades ──────────────────────────
    pos_np    = np.asarray(jax.device_get(pos_jax)) % mesh_size
    feats_np  = np.asarray(jax.device_get(feats_jax))   # [N, 9] strain E = D-I
    det_D_np  = np.asarray(jax.device_get(det_D_jax))   # [N] Jacobian det
    df_np     = np.asarray(jax.device_get(delta_f_jax)) # [N, 3] fuerza target

    # Magnitudes (Frobenius para E, L2 para fuerza)
    strain_mag = np.sqrt(np.sum(feats_np[:, :9]**2, axis=-1))
    force_mag  = np.sqrt(np.sum(df_np**2, axis=-1))
    
    # ── 2. Máscara de Sub-volumen ──────────────────────────────────────────
    SV_CENTER = np.array(sv_center)
    in_sv = np.all(np.abs(pos_np - SV_CENTER) < sv_half, axis=1)
    sv_idx = np.where(in_sv)[0]
    
    if len(sv_idx) == 0:
        print("⚠️ Sub-volumen vacío.")
        return

    # Selección aleatoria para destacar partículas centrales
    rng = np.random.default_rng(42)
    sel_idx = rng.choice(sv_idx, size=min(n_sel, len(sv_idx)), replace=False)

    # ── 3. Configuración de Paneles ─────────────────────────────────────────
    quantities = [
        (strain_mag, r"$\|E\|_F$ magnitud de deformación", "YlOrRd", False),
        (det_D_np,   r"$\det(D)$ Jacobiano (< 0 = shell-crossing)", "RdBu_r", True),
        (force_mag,  r"$|\Delta F|$ magnitud del target", "magma", False),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), facecolor="#0D1117")

    for ax, (qty, title, cmap, symm) in zip(axes, quantities):
        sq = qty[sv_idx]
        lo_ = np.percentile(sq, 2)
        hi_ = np.percentile(sq, 98)
        
        if symm: # Centrar colormap en 0 para det(D)
            lim = max(abs(lo_), abs(hi_))
            lo_, hi_ = -lim, lim

        # Scatter principal del sub-volumen
        sc = ax.scatter(
            pos_np[sv_idx, 0], pos_np[sv_idx, 1],
            c=sq, s=1.5, cmap=cmap, vmin=lo_, vmax=hi_,
            rasterized=True
        )

        # Destacar partículas centrales seleccionadas
        ax.scatter(
            pos_np[sel_idx, 0], pos_np[sel_idx, 1],
            s=18, c="white", edgecolors="#AAAAAA", linewidths=0.3, zorder=5
        )

        # Resaltar puntos con Shell-Crossing (det D < 0)
        if symm:
            sc_mask = (det_D_np[sv_idx] < 0)
            if sc_mask.any():
                ax.scatter(
                    pos_np[sv_idx[sc_mask], 0], pos_np[sv_idx[sc_mask], 1],
                    s=6, marker="x", c="lime", linewidths=0.5, alpha=0.6,
                    zorder=4, label=f"SC ({sc_mask.mean():.1%})"
                )
                ax.legend(fontsize=7, loc="upper right", framealpha=0.3,
                          labelcolor="white").get_frame().set_facecolor("#1A1A2E")

        # Colorbar estilizada
        cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.yaxis.set_tick_params(color="white", labelsize=7)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

        # Estética de ejes
        ax.set_facecolor("#0D1117")
        ax.set_title(title, color="white", fontsize=9, pad=8)
        ax.set_xlabel("x [mesh cells]", color="white", fontsize=8)
        ax.set_ylabel("y [mesh cells]", color="white", fontsize=8)
        ax.tick_params(colors="white", labelsize=7)
        ax.set_aspect("equal")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333344")

    fig.suptitle(
        f"Análisis de Features Lagrangianas | a = {a_val:.3f} | "
        f"SC Global: {float(np.mean(det_D_np < 0)):.2%}",
        color="white", fontsize=11, y=1.01
    )
    plt.tight_layout()
    plt.show()

    # ── 4. Diagnóstico de Señal y Correlaciones ───────────────────────────
    r_strain_force, p_val = pearsonr(strain_mag[sv_idx], force_mag[sv_idx])
    # Correlación de det(D) con fuerza (usamos |det D - 1| como medida de desviación de la linealidad)
    r_det_force, _ = pearsonr(np.abs(det_D_np[sv_idx] - 1.0), force_mag[sv_idx])

    print("\n── Correlaciones en sub-volumen (Pearson) ───────────────")
    print(f"  R( ||E||_F , |ΔF| ) = {r_strain_force:+.4f}  p={p_val:.2e}")
    print(f"  R( |det D - 1| , |ΔF| ) = {r_det_force:+.4f}")
    
    verdict = "FUERTE ✓" if abs(r_strain_force) > 0.3 else "DÉBIL (Revisar features)"
    print(f"  → Señal Lagrangiana detectada: {verdict}")

    print("\n── Estadísticas del Sub-volumen ────────────────────────")
    print(f"  Partículas totales : {len(sv_idx):,}")
    print(f"  Shell-crossing (SC): {int(np.sum(det_D_np[sv_idx]<0))} ({np.mean(det_D_np[sv_idx]<0):.2%})")
    print(f"  ||E||_F mean ± std : {strain_mag[sv_idx].mean():.4f} ± {strain_mag[sv_idx].std():.4f}")
    print(f"  |ΔF|     mean ± std : {force_mag[sv_idx].mean():.4f} ± {force_mag[sv_idx].std():.4f}")

# Ejemplo de uso:
# plot_lagrangian_features_analysis(pos_lr_t, feats_train, det_D_train, delta_f_tr, mesh_lr, a_train)

def plot_feature_force_correlation(feats_jax, delta_f_jax, a_val, 
                                   use_invariants=False, 
                                   sim_id=0, 
                                   snap_id=24):
    """
    Calcula y visualiza la matriz de correlación de Pearson entre los componentes
    del tensor de deformación (y sus invariantes) con los componentes de la fuerza.
    """
    # 1. Preparación de datos
    feats_np = np.asarray(jax.device_get(feats_jax))   # [N, 9]
    delta_np = np.asarray(jax.device_get(delta_f_jax)) # [N, 3]
    
    strain_labels = [
        "E_xx", "E_xy", "E_xz",
        "E_yx", "E_yy", "E_yz",
        "E_zx", "E_zy", "E_zz",
    ]

    # 2. Cálculo opcional de invariantes
    if use_invariants:
        # Re-formatear a [N, 3, 3] para cálculos matriciales
        E = feats_np[:, :9].reshape(-1, 3, 3)
        I = np.eye(3)
        D = E + I  # Tensor de deformación completo D = E + I
        
        tr_E = E[:, 0, 0] + E[:, 1, 1] + E[:, 2, 2]
        det_D = np.linalg.det(D)
        frob_E = np.sqrt(np.sum(E**2, axis=(1, 2)))
        
        # Concatenar a los features
        invariants = np.stack([tr_E, det_D, frob_E], axis=1)
        feats_np = np.concatenate([feats_np, invariants], axis=1)
        strain_labels += ["tr(E)", "det(D)", "||E||_F"]

    # 3. Calcular matriz de correlación
    n_feats = feats_np.shape[1]
    r_matrix = np.zeros((n_feats, 3))
    
    for fi in range(n_feats):
        for fc in range(3):
            # Ignorar NaNs o constantes para evitar errores en pearsonr
            if np.std(feats_np[:, fi]) > 0 and np.std(delta_np[:, fc]) > 0:
                r_matrix[fi, fc], _ = pearsonr(feats_np[:, fi], delta_np[:, fc])

    # 4. Graficado
    fig, ax = plt.subplots(figsize=(8, 7), facecolor='white')
    im = ax.imshow(r_matrix, cmap="RdBu", vmin=-0.6, vmax=0.6, aspect="auto")
    
    # Configuración de ejes
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([r"$\Delta F_x$", r"$\Delta F_y$", r"$\Delta F_z$"])
    ax.set_yticks(range(n_feats))
    ax.set_yticklabels(strain_labels, fontsize=10)
    
    plt.colorbar(im, ax=ax, label="Pearson R")
    ax.set_title(
        f"Feature–Force Correlation Matrix\n"
        f"sim={sim_id} | snap={snap_id} | a={a_val:.3f} | N={feats_np.shape[0]:,}",
        fontsize=12, pad=15
    )

    # Anotaciones numéricas en las celdas
    for fi in range(n_feats):
        for fc in range(3):
            val = r_matrix[fi, fc]
            ax.text(fc, fi, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, color="black" if abs(val) < 0.35 else "white")

    plt.tight_layout()
    plt.show()

    # 5. Diagnóstico por consola
    max_r = np.abs(r_matrix).max()
    print(f"── Diagnóstico de Señal ──────────────────────────")
    print(f"  Max |R| detectado: {max_r:.4f}")
    
    if max_r > 0.30:
        print("  → RESULTADO: SEÑAL PRESENTE ✓ (El tensor de deformación explica el residual)")
    elif max_r > 0.10:
        print("  → RESULTADO: SEÑAL DÉBIL (Patrones no lineales, el MLP será necesario)")
    else:
        print("  → RESULTADO: SIN SEÑAL (Revisar el cálculo de las features o el snapshot)")
    print(f"──────────────────────────────────────────────────\n")

# Ejemplo de uso:
# plot_feature_force_correlation(feats_train, delta_f_tr, a_train, use_invariants=True)

def plot_force_distribution_analysis(force_mag, det_D, strain_mag, a_val, 
                                     n_subsample=10_000, 
                                     seed=0):
    """
    Compara la distribución de los residuales de fuerza entre partículas con y sin 
    shell-crossing, y muestra la relación entre deformación y fuerza.
    """
    # 1. Asegurar formatos numpy
    fm = np.asarray(force_mag)
    dD = np.asarray(det_D)
    sm = np.asarray(strain_mag)
    
    # 2. Máscaras de Shell-Crossing
    sc_mask = dD < 0
    normal_mask = ~sc_mask
    sc_frac = np.mean(sc_mask)
    
    # Coeficiente de correlación global para el título
    r_sf, _ = pearsonr(sm, fm)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Panel Izquierdo: Histogramas de Densidad ---
    # Usamos el percentil 99 para evitar que outliers estiren demasiado el eje X
    x_max = np.percentile(fm, 99)
    bins = np.linspace(0, x_max, 60)
    
    axes[0].hist(fm[normal_mask], bins=bins, density=True, alpha=0.6, 
                 label=f"Normal (det>0), μ={fm[normal_mask].mean():.2f}", color="C0")
    axes[0].hist(fm[sc_mask], bins=bins, density=True, alpha=0.6, 
                 label=f"Shell-crossing (det<0), μ={fm[sc_mask].mean():.2f}", color="C1")
    
    # Líneas de tendencia (medias)
    axes[0].axvline(fm[normal_mask].mean(), color="C0", ls="--", lw=1.5)
    axes[0].axvline(fm[sc_mask].mean(), color="C1", ls="--", lw=1.5)
    
    axes[0].set_xlabel(r"$|\Delta F|$ [mesh units]")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Force residual distribution by status")
    axes[0].legend(fontsize=9)

    # --- Panel Derecho: Scatter Plot de Deformación vs Fuerza ---
    rng = np.random.default_rng(seed)
    # Submuestreo para evitar saturación y lentitud al graficar
    ss_indices = rng.choice(len(fm), size=min(n_subsample, len(fm)), replace=False)
    
    # Partículas normales en el submuestreo
    ss_normal = ss_indices[normal_mask[ss_indices]]
    axes[1].scatter(sm[ss_normal], fm[ss_normal], s=2, alpha=0.3, 
                    c="steelblue", label="Normal", rasterized=True)
    
    # Partículas SC en el submuestreo (las resaltamos un poco más)
    ss_sc = ss_indices[sc_mask[ss_indices]]
    axes[1].scatter(sm[ss_sc], fm[ss_sc], s=6, alpha=0.6, 
                    c="tomato", label="Shell-crossing", rasterized=True)
    
    axes[1].set_xlabel(r"$\|E\|_F$ (strain magnitude)")
    axes[1].set_ylabel(r"$|\Delta F|$ [mesh units]")
    axes[1].set_title(f"Strain vs Force Residual (R={r_sf:.3f})")
    axes[1].legend(markerscale=3, fontsize=9)

    plt.suptitle(f"Force Residual Analysis | a={a_val:.3f} | Global SC frac={sc_frac:.2%}", 
                 fontsize=12, y=1.02)
    
    plt.tight_layout()
    plt.show()

    # Diagnóstico rápido en consola
    print(f"── Estadísticas por Población ──────────────────")
    print(f"  Media |ΔF| (Normal): {fm[normal_mask].mean():.4f}")
    if sc_frac > 0:
        print(f"  Media |ΔF| (SC)    : {fm[sc_mask].mean():.4f}")
        diff = (fm[sc_mask].mean() / fm[normal_mask].mean() - 1) * 100
        print(f"  → El error es {diff:+.1f}% mayor en regiones de Shell-Crossing")
    print(f"────────────────────────────────────────────────\n")

# Ejemplo de uso:
# plot_force_distribution_analysis(force_mag, det_D_train, strain_mag, a_train)


def plot_prediction_vs_target_hex(delta_f_jax, f_pred_jax, a_val, 
                                   sim_val=0, snap_val=24, 
                                   r_mean=0, mse_tot=0, frac_mse=0,
                                   gridsize=70):
    """
    Compara las componentes (x, y, z) de la fuerza predicha vs el target 
    usando hexbins con escala logarítmica para manejar la densidad de puntos.
    """
    # 1. Preparación de datos (pasar a CPU)
    tgt_np = np.asarray(jax.device_get(delta_f_jax))
    prd_np = np.asarray(jax.device_get(f_pred_jax))
    
    comps = ["x", "y", "z"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor='white')

    for ci, ax in enumerate(axes):
        tgt = tgt_np[:, ci]
        prd = prd_np[:, ci]

        # 2. Definir límites basados en percentiles para evitar outliers
        lim = max(abs(np.percentile(tgt, 1)), abs(np.percentile(tgt, 99)))
        lim *= 1.15

        # 3. Hexbin con escala logarítmica en la densidad (C=None para contar puntos)
        hb = ax.hexbin(tgt, prd, gridsize=gridsize, cmap="plasma", 
                       norm=LogNorm(), mincnt=1,
                       extent=[-lim, lim, -lim, lim])
        
        cb = plt.colorbar(hb, ax=ax)
        cb.set_label("N particles", fontsize=9)

        # Línea de identidad (Ideal)
        ax.plot([-lim, lim], [-lim, lim], "w--", lw=1.5, alpha=0.8, label="Ideal")

        # 4. Estadística local
        r_c, _ = pearsonr(tgt, prd)
        
        ax.set_xlabel(f"Target $\Delta F_{{{comps[ci]}}}$")
        ax.set_ylabel(f"Predicted $\Delta F_{{{comps[ci]}}}$")
        ax.set_title(f"Component {comps[ci].upper()}\n$R = {r_c:.4f}$")
        # ax.set_aspect("equal")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, loc="upper left")

    # 5. Título global con métricas del modelo
    plt.suptitle(
        f"Force Residual: Predicted vs Target — sim={sim_val} snap={snap_val} a={a_val:.3f}\n"
        f"$\\bar{{R}}$ = {r_mean:.4f}  |  MSE = {mse_tot:.3e}  |  FracMSE = {frac_mse:.3f}",
        fontsize=12, y=1.02
    )

    plt.tight_layout()
    plt.show()

# Ejemplo de uso:
# plot_prediction_vs_target_hex(delta_f_tr, f_pred_np, a_train, 
#                               sim_val=SIM_VAL, snap_val=SNAP_VAL, 
#                               r_mean=r_mean, mse_tot=mse_tot, frac_mse=frac_mse)