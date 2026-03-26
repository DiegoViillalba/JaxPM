# ==============================================================================
# sweep.py  —  Barrido de hiperparámetros para JaxPM neural correction
#
# Uso:
#   python sweep.py                          # lanza todos los runs secuenciales
#   python sweep.py --dry_run                # imprime configs sin entrenar
#   python sweep.py --group mi_experimento   # tag de WandB para agrupar runs
#   python sweep.py --start_idx 3            # reanuda desde el run #3
#
# Diseño:
#   - Convergencia esperada ~50 epochs → n_steps=60 con patience=15
#   - Barrido cubre los ejes que mostraron más varianza en tus runs previos:
#     model type, lambda_pk, channels, lr, weight_snapshots
#   - Cada run es independiente (subprocess) para garantizar que JAX libere
#     GPU memory completamente entre runs (no hay forma de hacer esto in-process)
#   - Resultados se consolidan en sweep_results.jsonl al final de cada run
# ==============================================================================

import os
import sys
import json
import time
import argparse
import itertools
import subprocess
import tempfile
from copy import deepcopy
from pathlib import Path
from datetime import datetime

import yaml

# ── Rutas — ajusta si es necesario ───────────────────────────────────────────
TRAIN_SCRIPT = Path("/cosmos_storage/home/diegovillalba/JaxPM/pm2nbody/train_refactored_new.py")
DATA_DIR     = Path("/cosmos_storage/home/diegovillalba/JaxPM/data/")
MODELS_DIR   = Path("/cosmos_storage/home/diegovillalba/JaxPM/models/")
SWEEP_DIR    = MODELS_DIR / f"sweep_{datetime.now().strftime('%Y%m%d_%H%M')}"

# ── Base config — hereda todo de electric-dream, solo se pisan las claves del grid
BASE_CONFIG = {
    "data": {
        "mesh_lr": 128,
        "mesh_hr": 256,
        "n_train_sims": 40,
        "n_val_sims": 1,
        "n_test_sims": 1,
        "box_size": 256.0,
        "n_snapshots": 50,
        "n_particles": 128,
        "snapshots": None,
    },
    "correction_model": {
        "type": "cnn",
        "channels_hidden_dim": 16,
        "n_convolutions": 3,
        "n_fully_connected": 2,
        "input_dim": 1,
        "kernel_size": 3,
        "pad_periodic": True,
        "embed_globals": False,
        "n_globals_embedding": 1,
        "globals_embedding_dim": 64,
        "global_conditioning": "add",
        "use_attention_interpolation": False,
        "add_particle_velocities": True,   # clave en electric-dream
        "n_knots": 16,
        "latent_size": 64,
    },
    "training": {
        "seed": 0,
        "n_steps": 60,           # convergencia ~50 epochs → 60 con margen
        "batch_size": 1,
        "patience": 15,          # early stop proporcional al presupuesto
        "checkpoint_every": 10,
        "sample_snapshots": False,
        "loss": "mse_positions",
        "weight_snapshots": True,
        "lambda_pos": 1.0,
        "lambda_velocity": 1.0,
        "lambda_density": 0.0,
        "lambda_pk": 0.0,
        "lambda_cross_corr": 0.0,
        "log_pos": False,
        "fractional_mse": False,
        "weight_decay": 1e-4,
        "max_idx": 49,
        "schedule": {
            "type": "cosine",
            "initial_lr": 0.0,
            "peak_value": 4e-4,
            "warmup_steps": 5,
            "n_steps": 60,
            "factor": 0.5,
            "patience": 5,
            "min_lr": 5e-5,
        },
    },
    "wandb": {"project": "pm2nbody"},
}

# ==============================================================================
# Grid de hiperparámetros
# Basado en el análisis de tus runs:
#   - kcorr es el modelo más estable → vale la pena explorar sus knots/latent
#   - CNN mejora con lambda_pk > 0 para atacar over-smoothing
#   - channels_hidden_dim y peak_lr son los más sensibles en el MSE
#   - add_particle_velocities fue clave en electric-dream → lo dejamos fijo en True
# ==============================================================================

SWEEP_GRID = {

    # ── Eje 1: tipo de modelo + configuración espectral ──────────────────────
    # Pregunta: ¿cuánto ayuda lambda_pk y cómo interactúa con el tipo de modelo?
    "model_type_x_pk": [
        {"correction_model.type": "cnn",   "training.lambda_pk": 0.0},   # baseline electric-dream
        {"correction_model.type": "cnn",   "training.lambda_pk": 0.05},  # pk loss suave
        {"correction_model.type": "cnn",   "training.lambda_pk": 0.1},   # pk loss moderado
        {"correction_model.type": "cnn",   "training.lambda_pk": 0.2},   # pk loss agresivo
        {"correction_model.type": "kcorr", "training.lambda_pk": 0.0},   # kcorr baseline
        {"correction_model.type": "kcorr", "training.lambda_pk": 0.05},  # kcorr + pk
    ],

    # ── Eje 2: capacidad de la red CNN ───────────────────────────────────────
    # Pregunta: ¿el modelo está underfitting o ya saturó con 16 canales?
    "channels": [
        {"correction_model.channels_hidden_dim": 8,  "correction_model.n_convolutions": 3},
        {"correction_model.channels_hidden_dim": 16, "correction_model.n_convolutions": 3},  # baseline
        {"correction_model.channels_hidden_dim": 32, "correction_model.n_convolutions": 3},
        {"correction_model.channels_hidden_dim": 16, "correction_model.n_convolutions": 5},  # más profundo
    ],

    # ── Eje 3: learning rate peak ────────────────────────────────────────────
    # Pregunta: ¿el lr de electric-dream es óptimo o hay margen?
    "peak_lr": [
        {"training.schedule.peak_value": 1e-4,  "training.schedule.min_lr": 1e-5},
        {"training.schedule.peak_value": 4e-4,  "training.schedule.min_lr": 5e-5},  # baseline
        {"training.schedule.peak_value": 8e-4,  "training.schedule.min_lr": 1e-4},
        {"training.schedule.peak_value": 1.5e-3,"training.schedule.min_lr": 2e-4},
    ],

    # ── Eje 4: pesos de snapshot + velocidad ────────────────────────────────
    # Pregunta: ¿weight_snapshots y lambda_velocity mejoran estabilidad temporal?
    "snapshot_vel_weights": [
        {"training.weight_snapshots": True,  "training.lambda_velocity": 1.0},  # baseline
        {"training.weight_snapshots": True,  "training.lambda_velocity": 0.5},
        {"training.weight_snapshots": False, "training.lambda_velocity": 1.0},
        {"training.weight_snapshots": True,  "training.lambda_velocity": 2.0},
    ],

    # ── Eje 5: combinación cnn+kcorr ─────────────────────────────────────────
    # Pregunta: ¿el pipeline combinado supera a cada modelo por separado?
    "combined": [
        {"correction_model.type": "cnn+kcorr", "training.lambda_pk": 0.0},
        {"correction_model.type": "cnn+kcorr", "training.lambda_pk": 0.05},
    ],
}


# ==============================================================================
# Utilidades
# ==============================================================================

def set_nested(d: dict, dotkey: str, value):
    """Set d['a']['b']['c'] from dotkey='a.b.c'."""
    keys = dotkey.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def build_config(overrides: dict) -> dict:
    """Deep-copy base config and apply flat dotkey overrides."""
    cfg = deepcopy(BASE_CONFIG)
    for dotkey, value in overrides.items():
        set_nested(cfg, dotkey, value)
    # Keep schedule.n_steps in sync with training.n_steps
    cfg["training"]["schedule"]["n_steps"] = cfg["training"]["n_steps"]
    return cfg


def config_to_tag(overrides: dict) -> str:
    """Short human-readable tag from overrides for logging."""
    parts = []
    for k, v in overrides.items():
        short_k = k.split(".")[-1]
        parts.append(f"{short_k}={v}")
    return " | ".join(parts)


def flatten_runs() -> list[tuple[str, dict]]:
    """Returns list of (axis_name, overrides_dict) for all runs."""
    runs = []
    for axis_name, override_list in SWEEP_GRID.items():
        for overrides in override_list:
            runs.append((axis_name, overrides))
    return runs


def launch_run(cfg: dict, run_idx: int, total: int, tag: str, group: str, dry_run: bool) -> dict:
    """
    Write cfg to a temp YAML and launch train_refactored.py as a subprocess.
    Returns a result dict with timing and exit code.
    """
    print(f"\n{'='*64}")
    print(f"  Run {run_idx+1}/{total}  |  {tag}")
    print(f"{'='*64}")

    if dry_run:
        print("  [DRY RUN] Config:")
        print(yaml.dump(cfg, default_flow_style=False, indent=2))
        return {"run_idx": run_idx, "tag": tag, "status": "dry_run", "duration_s": 0}

    # Inject sweep group into wandb config so runs are grouped in the dashboard
    cfg["wandb"]["group"] = group

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, prefix=f"sweep_run{run_idx}_"
    ) as f:
        yaml.dump(cfg, f)
        tmp_path = f.name

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT), f"--config={tmp_path}"],
        timeout=7200,   # 2h max per run
    )
    duration = time.time() - t0

    os.unlink(tmp_path)  # clean up temp config

    status = "ok" if result.returncode == 0 else f"failed (exit {result.returncode})"
    print(f"\n  → {status}  |  {duration/60:.1f} min")

    return {
        "run_idx":    run_idx,
        "tag":        tag,
        "overrides":  cfg,
        "status":     status,
        "duration_s": round(duration, 1),
        "returncode": result.returncode,
    }


# ==============================================================================
# Entry point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="JaxPM hyperparameter sweep")
    parser.add_argument("--dry_run",   action="store_true",
                        help="Print configs without training")
    parser.add_argument("--group",     type=str, default=None,
                        help="WandB group tag (default: sweep_YYYYMMDD_HHMM)")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Skip runs before this index (for resuming)")
    parser.add_argument("--axis",       type=str, default=None,
                        help="Run only one axis of the grid (e.g. 'channels')")
    parser.add_argument("--no_confirm", action="store_true",
                        help="Skip confirmation prompt (required for nohup/batch)")
    args = parser.parse_args()

    group = args.group or f"sweep_{datetime.now().strftime('%Y%m%d_%H%M')}"
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)

    all_runs = flatten_runs()

    # Filter to a single axis if requested
    if args.axis:
        all_runs = [(ax, ov) for ax, ov in all_runs if ax == args.axis]
        if not all_runs:
            print(f"ERROR: axis '{args.axis}' not found. Available: {list(SWEEP_GRID.keys())}")
            sys.exit(1)

    total     = len(all_runs)
    results   = []
    results_f = SWEEP_DIR / "sweep_results.jsonl"

    print(f"\nSweep group : {group}")
    print(f"Total runs  : {total}  (starting from idx {args.start_idx})")
    print(f"Output dir  : {SWEEP_DIR}")
    print(f"Results log : {results_f}\n")

    print("Run plan:")
    for i, (axis, overrides) in enumerate(all_runs):
        marker = "→" if i >= args.start_idx else "✓"
        print(f"  [{marker}] {i:02d}  [{axis}]  {config_to_tag(overrides)}")

    if not args.dry_run and not args.no_confirm:
        try:
            confirm = input("\nProceed? [y/N] ").strip().lower()
        except OSError:
            # stdin not available (nohup, batch) — require explicit --no_confirm
            print("\nERROR: no stdin available. Re-run with --no_confirm to skip this prompt.")
            sys.exit(1)
        if confirm != "y":
            print("Aborted.")
            sys.exit(0)

    for i, (axis_name, overrides) in enumerate(all_runs):
        if i < args.start_idx:
            print(f"  Skipping run {i} (--start_idx={args.start_idx})")
            continue

        tag = f"[{axis_name}] {config_to_tag(overrides)}"
        cfg = build_config(overrides)

        result = launch_run(
            cfg=cfg,
            run_idx=i,
            total=total,
            tag=tag,
            group=group,
            dry_run=args.dry_run,
        )
        result["axis"] = axis_name
        results.append(result)

        # Append to JSONL incrementally so partial sweeps are recoverable
        with open(results_f, "a") as f:
            f.write(json.dumps(result) + "\n")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print(f"  Sweep complete  —  {sum(r['status']=='ok' for r in results)}/{len(results)} OK")
    print(f"{'='*64}")
    for r in results:
        icon = "✓" if r["status"] == "ok" else "✗"
        mins = r["duration_s"] / 60
        print(f"  {icon}  {r['run_idx']:02d}  [{r.get('axis','')}]  {r['tag']}"
              f"  ({mins:.1f} min)")

    print(f"\nResults saved → {results_f}")


if __name__ == "__main__":
    main()