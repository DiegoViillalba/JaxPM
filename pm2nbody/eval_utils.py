"""
eval_utils.py — Herramientas de evaluación para PM2Nbody.

Punto central de dispatch por tipo de modelo.  El notebook de evaluación
actúa como orquestador puro; toda la lógica pesada vive aquí.

Funciones públicas principales
-------------------------------
verify_runs(runs)              — imprime tabla de estado de las corridas
load_run(run_dir, checkpoint)  — lee config.yaml y devuelve {model, params, …}
load_sim(...)                  — carga una simulación en unidades mesh_lr
run_ode(...)                   — integra la ODE con corrección neuronal
compute_sim_metrics(...)       — P(k)/P_HR, r(k), MSE de posiciones
aggregate_metrics(results, …)  — media ± std sobre sims de test
slab_projection(...)           — proyección 2D de densidad log(1+δ)
save_results(...)              — guarda CSV, NPY y JSON de resultados
"""

import pickle
import yaml
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Union

import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk

from jaxpm.pm import make_ode_fn, get_delta
from jaxpm.nn import CNN, NeuralSplineFourierFilter
from jaxpm.painting import compensate_cic
from jaxpm.utils import power_spectrum, cross_correlation_coefficients


# ==============================================================================
# 1. Model building — dispatch por type
# ==============================================================================

def build_model_from_config(cm: dict):
    """
    Construye el modelo Haiku correcto según el campo `type` de config.yaml.

    Soportado:
      cnn / cnn_force / kcorr / cnn+kcorr / cnn_wst / patched_transformer

    Los parámetros opcionales se rellenan con defaults razonables cuando
    la corrida no los especificó explícitamente (compatibilidad con runs viejas).
    """
    model_type = cm["type"]

    if model_type in ("cnn", "cnn_force"):
        output_dim = 3 if model_type == "cnn_force" else 1

        def CNNCorr(x, positions, scale_factors, velocities):
            cnn = CNN(
                channels_hidden_dim=cm["channels_hidden_dim"],
                n_convolutions=cm["n_convolutions"],
                n_fully_connected=cm["n_fully_connected"],
                input_dim=cm.get("input_dim", 2),
                output_dim=output_dim,
                kernel_size=cm.get("kernel_size", 3),
                pad_periodic=cm.get("pad_periodic", True),
                embed_globals=cm.get("embed_globals", False),
                n_globals_embedding=cm.get("n_globals_embedding", 1),
                globals_embedding_dim=cm.get("globals_embedding_dim", 64),
                global_conditioning=cm.get("global_conditioning", "add"),
                use_attention_interpolation=cm.get("use_attention_interpolation", False),
                add_particle_velocities=cm.get("add_particle_velocities", True),
            )
            return cnn(
                x=x, positions=positions,
                global_features=scale_factors, velocities=velocities,
            )

        return hk.without_apply_rng(hk.transform(CNNCorr))

    elif model_type == "kcorr":
        def KCorr(x, scale_factors):
            return NeuralSplineFourierFilter(
                n_knots=cm.get("n_knots", 16),
                latent_size=cm.get("latent_size", 64),
            )(x, scale_factors)

        return hk.without_apply_rng(hk.transform(KCorr))

    elif model_type == "cnn+kcorr":
        def CNNCorr(x, positions, scale_factors, velocities):
            cnn = CNN(
                channels_hidden_dim=cm["channels_hidden_dim"],
                n_convolutions=cm["n_convolutions"],
                n_fully_connected=cm["n_fully_connected"],
                input_dim=cm.get("input_dim", 2),
                output_dim=1,
                kernel_size=cm.get("kernel_size", 3),
                pad_periodic=cm.get("pad_periodic", True),
                embed_globals=cm.get("embed_globals", False),
                n_globals_embedding=cm.get("n_globals_embedding", 1),
                globals_embedding_dim=cm.get("globals_embedding_dim", 64),
                global_conditioning=cm.get("global_conditioning", "add"),
                use_attention_interpolation=cm.get("use_attention_interpolation", False),
                add_particle_velocities=cm.get("add_particle_velocities", True),
            )
            return cnn(
                x=x, positions=positions,
                global_features=scale_factors, velocities=velocities,
            )

        def KCorr(x, scale_factors):
            return NeuralSplineFourierFilter(
                n_knots=cm.get("n_knots", 16),
                latent_size=cm.get("latent_size", 64),
            )(x, scale_factors)

        return {
            "cnn":   hk.without_apply_rng(hk.transform(CNNCorr)),
            "kcorr": hk.without_apply_rng(hk.transform(KCorr)),
        }

    elif model_type == "cnn_wst":
        from jaxpm.wst import WaveletScatteringTransform
        J = int(cm["wst_J"])
        L = int(cm["wst_L"])

        def CNNWSTCorr(x, positions, scale_factors, velocities):
            wst   = WaveletScatteringTransform(J=J, L=L, normalize=True, name="wst")
            x_wst = wst(x)
            actual_input_dim = x.shape[-1] + J * L
            cnn = CNN(
                channels_hidden_dim=cm["channels_hidden_dim"],
                n_convolutions=cm["n_convolutions"],
                n_fully_connected=cm["n_fully_connected"],
                input_dim=actual_input_dim,
                output_dim=1,
                kernel_size=cm.get("kernel_size", 3),
                pad_periodic=cm.get("pad_periodic", True),
                embed_globals=cm.get("embed_globals", False),
                n_globals_embedding=cm.get("n_globals_embedding", 1),
                globals_embedding_dim=cm.get("globals_embedding_dim", 64),
                global_conditioning=cm.get("global_conditioning", "add"),
                use_attention_interpolation=cm.get("use_attention_interpolation", False),
                add_particle_velocities=cm.get("add_particle_velocities", True),
            )
            return cnn(
                x=x_wst, positions=positions,
                global_features=scale_factors, return_features=False,
                velocities=velocities,
            )

        return hk.without_apply_rng(hk.transform(CNNWSTCorr))

    elif model_type == "patched_transformer":
        from jaxpm.patched_transformer import make_patched_transformer
        return make_patched_transformer(
            J=cm.get("wst_J", 3),
            L=cm.get("wst_L", 4),
            patch_size=cm.get("patch_size", 8),
            K=cm.get("K", 64),
            D_embed=cm.get("D_embed", 16),
            D_hidden=cm.get("D_hidden", 32),
            D_trans=cm.get("D_trans", 8),
            n_mlp_layers=cm.get("n_mlp_layers", 2),
            grid_size=cm.get("grid_size", 128),
        )

    elif model_type == "hybrid_transformer":
        from jaxpm.patched_transformer import make_hybrid_transformer
        return make_hybrid_transformer(
            J=cm.get("wst_J", 3),
            L=cm.get("wst_L", 4),
            patch_size=cm.get("patch_size", 8),
            K=cm.get("K", 64),
            D_embed=cm.get("D_embed", 16),
            D_hidden=cm.get("D_hidden", 32),
            D_trans=cm.get("D_trans", 8),
            n_mlp_layers=cm.get("n_mlp_layers", 2),
            grid_size=cm.get("grid_size", 128),
            combine_mode=cm.get("combine_mode", "sum"),
        )

    else:
        raise ValueError(
            f"Tipo de modelo desconocido: {model_type!r}. "
            f"Tipos soportados: cnn, cnn_force, kcorr, cnn+kcorr, cnn_wst, patched_transformer, hybrid_transformer"
        )


def _ode_correction_type(model_type: str) -> str:
    """
    Mapea el type de config.yaml al string add_correction de make_ode_fn.

    cnn_wst  → "cnn"  (la interfaz del modelo es idéntica al CNN estándar)
    todos los demás → igual que model_type
    """
    if model_type == "cnn_wst":
        return "cnn"
    return model_type


# ==============================================================================
# 2. Run loading
# ==============================================================================

def _resolve_run_entry(entry) -> dict:
    """
    Normaliza una entrada del registro RUNS a dict con campos:
      dir, checkpoint, display_label
    """
    if entry is None:
        return None
    if isinstance(entry, (str, Path)):
        return {"dir": Path(entry), "checkpoint": "best"}
    if isinstance(entry, dict):
        d = dict(entry)
        d["dir"] = Path(d["dir"])
        d.setdefault("checkpoint", "best")
        return d
    raise TypeError(f"Entrada RUNS inválida: {type(entry)}")


def load_run(run_dir: Union[Path, str, dict], checkpoint: str = "best") -> dict:
    """
    Lee config.yaml y carga el modelo + params para una corrida.

    Acepta:
      load_run(Path("models/mi-run"))                    — usa best*.pkl
      load_run({"dir": Path("..."), "checkpoint": "50"}) — usa *_50.pkl

    Devuelve dict con:
      model     — modelo Haiku listo para .apply()
      params    — parámetros del checkpoint
      type      — string del tipo (de config.yaml)
      ode_type  — string para make_ode_fn (cnn_wst → "cnn", resto igual)
      config    — dict completo de correction_model
      run_dir   — Path
      pkl_file  — Path al checkpoint cargado
    """
    entry = _resolve_run_entry(run_dir)
    run_dir   = entry["dir"]
    checkpoint = entry.get("checkpoint", checkpoint)

    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.yaml no encontrado en {run_dir}")

    with open(cfg_path) as f:
        full_config = yaml.safe_load(f) or {}

    cm         = full_config.get("correction_model", {})
    model_type = cm["type"]

    if checkpoint == "best":
        pkl_files = sorted(run_dir.glob("best*.pkl"))
    else:
        pkl_files = sorted(run_dir.glob(f"*_{checkpoint}.pkl"))

    if not pkl_files:
        raise FileNotFoundError(
            f"Sin checkpoint en {run_dir} (checkpoint={checkpoint!r}). "
            f"Archivos pkl disponibles: {sorted(run_dir.glob('*.pkl'))}"
        )

    pkl_file = pkl_files[-1]
    with open(pkl_file, "rb") as f:
        params = pickle.load(f)

    model = build_model_from_config(cm)

    return {
        "model":    model,
        "params":   params,
        "type":     model_type,
        "ode_type": _ode_correction_type(model_type),
        "config":   cm,
        "run_dir":  run_dir,
        "pkl_file": pkl_file,
    }


def verify_runs(runs: dict) -> None:
    """
    Imprime una tabla de estado para todas las corridas del registro.

    Formato del registro (mismo que el notebook):
      {display_name: None | Path | {"dir": Path, "checkpoint": str}}
    """
    print(f"{'Corrida':<40} {'Estado':<14} {'type':<25} {'Checkpoint'}")
    print("─" * 105)

    for name, entry in runs.items():
        if entry is None:
            print(f"{name:<40} {'✓ baseline':<14} {'—':<25} —")
            continue

        try:
            e = _resolve_run_entry(entry)
        except TypeError as exc:
            print(f"{name:<40} {'✗ inválido':<14} {'?':<25} {exc}")
            continue

        run_dir = e["dir"]
        ckpt    = e.get("checkpoint", "best")
        cfg_file = run_dir / "config.yaml"

        if not cfg_file.exists():
            print(f"{name:<40} {'✗ sin config':<14} {'?':<25} —")
            continue

        with open(cfg_file) as f:
            cfg = yaml.safe_load(f) or {}
        mtype = cfg.get("correction_model", {}).get("type", "?")

        if ckpt == "best":
            pkls = sorted(run_dir.glob("best*.pkl"))
        else:
            pkls = sorted(run_dir.glob(f"*_{ckpt}.pkl"))

        status    = "✓ ok"       if pkls else "✗ sin pkl"
        ckpt_name = pkls[-1].name if pkls else "—"

        print(f"{name:<40} {status:<14} {mtype:<25} {ckpt_name}")

    print("─" * 105)


# ==============================================================================
# 3. Data loading
# ==============================================================================

def load_sim(data_dir: Path, sim_id: int,
             mesh_lr: int, mesh_hr: int, box_size: float):
    """
    Carga una simulación en unidades mesh_lr (mismo convenio que read_data.py).

    Los archivos en disco están en unidades físicas [0, box_size) Mpc/h.
    Conversión: pos_mesh = pos_phys / box_size * mesh_lr

    Devuelve:
      pos_lr, vel_lr   [T, N_lr, 3]  float32
      pos_hr, vel_hr   [T, N_hr, 3]  float32 (en unidades mesh_lr)
      avals            [T]           float32
    """
    data_dir = Path(data_dir)
    scale    = float(mesh_lr) / float(box_size)

    pos_lr = jnp.array(np.load(data_dir / f"pos_m{mesh_lr}_s{sim_id}.npy") * scale, dtype=jnp.float32)
    vel_lr = jnp.array(np.load(data_dir / f"vel_m{mesh_lr}_s{sim_id}.npy") * scale, dtype=jnp.float32)
    pos_hr = jnp.array(np.load(data_dir / f"pos_m{mesh_hr}_s{sim_id}.npy") * scale, dtype=jnp.float32)
    vel_hr = jnp.array(np.load(data_dir / f"vel_m{mesh_hr}_s{sim_id}.npy") * scale, dtype=jnp.float32)
    avals  = jnp.array(np.load(data_dir / "scale_factors.npy"),               dtype=jnp.float32)

    return pos_lr, vel_lr, pos_hr, vel_hr, avals


# ==============================================================================
# 4. Inference
# ==============================================================================

def run_ode(pos_lr_0, vel_lr_0, avals, model, params,
            ode_type: str, mesh_lr: int, cosmology):
    """
    Integra la ODE con corrección neuronal.

    ode_type: valor de _ode_correction_type() — llave para make_ode_fn.
    Devuelve pos_pm [T, N, 3] sin mod (el mod se aplica al hacer CIC deposit).
    """
    from jax.experimental.ode import odeint

    pos_pm, vel_pm = odeint(
        make_ode_fn(
            mesh_shape=(mesh_lr, mesh_lr, mesh_lr),
            add_correction=ode_type,
            model=model,
        ),
        [pos_lr_0, vel_lr_0],
        avals,
        cosmology,
        params,
        rtol=1e-5,
        atol=1e-5,
    )
    return pos_pm, vel_pm


# ==============================================================================
# 5. Metrics
# ==============================================================================

def _to_delta(pos, mesh_lr: int):
    return get_delta(jnp.mod(pos, mesh_lr), (mesh_lr, mesh_lr, mesh_lr))


def compute_pk_ratio(delta_pm, delta_hr, box_size: float):
    """P(k)_pm / P(k)_hr."""
    box  = np.array([box_size] * 3)
    kmin = np.pi / box_size
    dk   = 2 * np.pi / box_size
    k,  pk_pm = power_spectrum(compensate_cic(delta_pm), boxsize=box, kmin=kmin, dk=dk)
    _,  pk_hr = power_spectrum(compensate_cic(delta_hr),  boxsize=box, kmin=kmin, dk=dk)
    valid = (pk_hr > 0) & jnp.isfinite(pk_hr) & jnp.isfinite(pk_pm)
    ratio = jnp.where(valid, pk_pm / pk_hr, jnp.nan)
    return np.asarray(k), np.asarray(ratio)


def compute_cross_corr(delta_pm, delta_hr, box_size: float):
    """r(k) = P_cross / sqrt(P_pm · P_hr)."""
    box  = np.array([box_size] * 3)
    kmin = np.pi / box_size
    dk   = 2 * np.pi / box_size
    k, p_cross = cross_correlation_coefficients(
        compensate_cic(delta_hr), compensate_cic(delta_pm),
        boxsize=box, kmin=kmin, dk=dk,
    )
    _, pk_pm = power_spectrum(compensate_cic(delta_pm), boxsize=box, kmin=kmin, dk=dk)
    _, pk_hr = power_spectrum(compensate_cic(delta_hr),  boxsize=box, kmin=kmin, dk=dk)
    r = jnp.real(p_cross) / jnp.sqrt(jnp.clip(pk_pm * pk_hr, 1e-30))
    return np.asarray(k), np.asarray(r)


def compute_pos_mse(pos_pm, pos_hr, mesh_lr: int) -> float:
    """MSE de posiciones con distancia mínima imagen periódica, normalizado por N²."""
    n = float(mesh_lr)
    d = pos_pm - pos_hr
    d = d - n * jnp.round(d / n)
    return float(jnp.mean(jnp.sum(d ** 2, axis=-1)) / n ** 2)


def compute_sim_metrics(pos_pm_cpu: np.ndarray, pos_hr_cpu: np.ndarray,
                         snap_indices: List[int], avals_cpu: np.ndarray,
                         mesh_lr: int, box_size: float) -> dict:
    """
    Calcula métricas para todos los snapshots de UNA simulación.

    Inputs: CPU numpy arrays (ya descargados de GPU).
    Retorna dict con listas:
      pk_ratio, cross_corr, pos_mse, avals, k_bins
    """
    k_bins = None
    out = {"pk_ratio": [], "cross_corr": [], "pos_mse": [], "avals": []}

    for snap in snap_indices:
        p_pm = jnp.array(pos_pm_cpu[snap])
        p_hr = jnp.array(pos_hr_cpu[snap])

        d_pm = _to_delta(p_pm, mesh_lr)
        d_hr = _to_delta(p_hr, mesh_lr)

        k, ratio = compute_pk_ratio(d_pm, d_hr, box_size)
        _, r_k   = compute_cross_corr(d_pm, d_hr, box_size)
        mse      = compute_pos_mse(p_pm, jnp.mod(p_hr, mesh_lr), mesh_lr)

        out["pk_ratio"].append(np.asarray(ratio))
        out["cross_corr"].append(np.asarray(r_k))
        out["pos_mse"].append(float(mse))
        out["avals"].append(float(avals_cpu[snap]))

        if k_bins is None:
            k_bins = np.asarray(k)

        del p_pm, p_hr, d_pm, d_hr

    out["k_bins"] = k_bins
    return out


# ==============================================================================
# 6. Aggregation
# ==============================================================================

def aggregate_metrics(results: dict, snap_indices: List[int]) -> dict:
    """
    Media ± std de métricas sobre sims de test.

    results : {run_name: {sim_id: compute_sim_metrics(...)}}
    Devuelve: {run_name: {snap_idx: {"a", "pk_ratio_mean", "pk_ratio_std", …}}}
    """
    agg = {}
    for run_name, run_res in results.items():
        agg[run_name] = {}
        for i, snap in enumerate(snap_indices):
            all_ratio = np.stack([run_res[s]["pk_ratio"][i]   for s in run_res])
            all_corr  = np.stack([run_res[s]["cross_corr"][i]  for s in run_res])
            all_mse   = np.array([run_res[s]["pos_mse"][i]    for s in run_res])
            a_val     = list(run_res.values())[0]["avals"][i]

            agg[run_name][snap] = {
                "a":               a_val,
                "pk_ratio_mean":   np.nanmean(all_ratio, axis=0),
                "pk_ratio_std":    np.nanstd(all_ratio,  axis=0),
                "cross_corr_mean": np.nanmean(all_corr,  axis=0),
                "cross_corr_std":  np.nanstd(all_corr,   axis=0),
                "pos_mse_mean":    float(np.mean(all_mse)),
                "pos_mse_std":     float(np.std(all_mse)),
            }
    return agg


# ==============================================================================
# 7. Visualization helpers
# ==============================================================================

def slab_projection(pos_cpu: np.ndarray, src_mesh: int, dst_mesh: int,
                     slab_thickness: int = 16, z0_frac: float = 0.5) -> np.ndarray:
    """
    CIC deposit → log10(1+δ) en un slab 2D.

    pos_cpu : numpy [N, 3] en unidades src_mesh
    Retorna : float32 array (dst_mesh, dst_mesh)
    """
    scale = dst_mesh / src_mesh
    z0    = int(z0_frac * dst_mesh)
    half  = slab_thickness // 2

    pos_s = jnp.mod(jnp.array(pos_cpu, dtype=jnp.float32) * scale, dst_mesh)
    delta = get_delta(pos_s, (dst_mesh, dst_mesh, dst_mesh))
    d_np  = np.asarray(jax.device_get(delta))
    del pos_s, delta

    sl   = slice(max(z0 - half, 0), min(z0 + half, dst_mesh))
    proj = d_np[:, :, sl].sum(axis=-1)
    return np.log10(np.clip(1.0 + proj, 1e-4, None)).astype(np.float32)


def make_color_ls_map(run_names, palette=None):
    """
    Asigna color y linestyle a cada corrida de forma consistente.
    LR baseline → gris punteado.
    """
    if palette is None:
        palette = ["#185FA5", "#0F6E56", "#993C1D", "#533AB7",
                   "#C9820A", "#2E8B57", "#8B1A1A", "#6A0DAD"]

    color_map = {}
    ls_map    = {}
    palette_idx = 0

    for name in run_names:
        if "baseline" in name.lower() or name.strip() == "LR (baseline)":
            color_map[name] = "#888780"
            ls_map[name]    = ("--", 1.2)
        else:
            color_map[name] = palette[palette_idx % len(palette)]
            ls_map[name]    = ("-", 1.6)
            palette_idx += 1

    return color_map, ls_map


# ==============================================================================
# 8. Export
# ==============================================================================

def save_results(results: dict, agg: dict, k_bins: np.ndarray,
                 snap_indices: List[int], results_dir: Path,
                 timestamp: Optional[str] = None) -> dict:
    """
    Guarda resultados a disco:
      metrics_per_sim_<ts>.csv
      metrics_aggregated_<ts>.csv
      k_bins_<ts>.npy
      eval_results_<ts>.json

    Devuelve dict {label: Path} con rutas de los archivos guardados.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas requerido para guardar resultados.")

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    saved = {}

    # ── Per-sim flat CSV ──────────────────────────────────────────────────────
    rows = []
    for run_name, run_res in results.items():
        for sim_id, sm in run_res.items():
            for i, snap in enumerate(snap_indices):
                row = {
                    "run": run_name, "sim_id": sim_id,
                    "snap_idx": snap, "a": sm["avals"][i],
                    "pos_mse":  sm["pos_mse"][i],
                }
                for ki, kv in enumerate(k_bins):
                    row[f"pk_ratio_k{kv:.4f}"]   = float(sm["pk_ratio"][i][ki])
                    row[f"cross_corr_k{kv:.4f}"] = float(sm["cross_corr"][i][ki])
                rows.append(row)

    p = results_dir / f"metrics_per_sim_{timestamp}.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    saved["flat_csv"] = p

    # ── Aggregated CSV ────────────────────────────────────────────────────────
    K_REFS = [0.1, 0.2, 0.3, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0]
    rows_agg = []
    for run_name in agg:
        for snap in snap_indices:
            d   = agg[run_name][snap]
            row = {
                "run": run_name, "snap_idx": snap,
                "a": d["a"],
                "pos_mse_mean": d["pos_mse_mean"],
                "pos_mse_std":  d["pos_mse_std"],
            }
            for kref in K_REFS:
                ki = int(np.argmin(np.abs(k_bins - kref)))
                row[f"pk_ratio_mean_k{kref}"]   = float(d["pk_ratio_mean"][ki])
                row[f"pk_ratio_std_k{kref}"]    = float(d["pk_ratio_std"][ki])
                row[f"cross_corr_mean_k{kref}"] = float(d["cross_corr_mean"][ki])
                row[f"cross_corr_std_k{kref}"]  = float(d["cross_corr_std"][ki])
            rows_agg.append(row)

    p = results_dir / f"metrics_aggregated_{timestamp}.csv"
    pd.DataFrame(rows_agg).to_csv(p, index=False)
    saved["agg_csv"] = p

    # ── k_bins ────────────────────────────────────────────────────────────────
    p = results_dir / f"k_bins_{timestamp}.npy"
    np.save(p, k_bins)
    saved["k_bins"] = p

    # ── JSON ──────────────────────────────────────────────────────────────────
    export = {
        "k_bins": k_bins.tolist(),
        "snap_eval_indices": snap_indices,
        "runs": {},
    }
    for run_name in agg:
        export["runs"][run_name] = {}
        for snap in snap_indices:
            d = agg[run_name][snap]
            export["runs"][run_name][str(snap)] = {
                "a":               d["a"],
                "pk_ratio_mean":   d["pk_ratio_mean"].tolist(),
                "pk_ratio_std":    d["pk_ratio_std"].tolist(),
                "cross_corr_mean": d["cross_corr_mean"].tolist(),
                "cross_corr_std":  d["cross_corr_std"].tolist(),
                "pos_mse_mean":    d["pos_mse_mean"],
                "pos_mse_std":     d["pos_mse_std"],
            }

    p = results_dir / f"eval_results_{timestamp}.json"
    with open(p, "w") as f:
        json.dump(export, f, indent=2)
    saved["json"] = p

    for label, path in saved.items():
        print(f"  ✓ {label:<16} → {path}")

    return saved