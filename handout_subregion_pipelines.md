# Pipelines de Sub-Región — Handout
**JaxPM / pm2nbody** · 2026-06-02

---

## Contexto: ¿qué teníamos antes?

| Pipeline | Partículas | Corrección |
|----------|-----------|-----------|
| `train_lag_force` | N³ LR, malla fina vs gruesa | Resolución de fuerza |
| `train_cnn_massres` + `train_lag_massres` | 64³ LR vs 128³ HR (misma malla PM) | Resolución de masa |

Todos los pipelines anteriores usan **todas** las partículas en entrenamiento y evaluación.

---

## Pipeline 1 — Sub-región zoom (`train_subregion_force.py`)

### Concepto
Dado el contexto global LR + posiciones HR en una **sub-región espacial**, predecir la corrección de fuerza para las partículas HR de esa región.

```
F_total = F_LR@HR  +  ΔF_CNN  +  ΔF_MLP
```

| Componente | Descripción |
|-----------|-------------|
| `F_LR@HR` | Fuerza PM de la densidad LR, evaluada en **todas** las posiciones HR |
| `ΔF_CNN` | Corrección CNN sobre la malla LR global, evaluada en posiciones HR del parche |
| `ΔF_MLP` | Corrección MLP con features Lagrangianas en **resolución HR** |

### Diferencias clave vs. mass-res pipeline

| Aspecto | `train_lag_massres` | `train_subregion_force` |
|---------|--------------------|-----------------------|
| Par de fuerzas | Stride-subsample HR → mismas posiciones que LR | Evalúa en **todas** las HR |
| Features | Resolución LR (mesh_lr grid) | Resolución HR (`pos_hr × R`, mesh_hr grid) |
| Mini-batches | Snapshot completo | Parches Lagrangianos aleatorios de `patch_n³` |
| CNN evalúa en | Posiciones LR | Posiciones HR sub-celda (interpolación CIC) |

### Nueva función central
```python
# en train_lag_massres.py
compute_subregion_force_pair(pos_lr, pos_hr, mesh_lr, mesh_hr, sigma_hr, sigma_lr)
# → (f_lr_at_hr, f_hr, delta_f)   todas en [mesh_hr³, 3]  unidades mesh_lr
```

### Workflow
```bash
python pm2nbody/generate_data_subregion.py --mode test
# (opcional) python pm2nbody/train_cnn_massres.py --config configs/cnn_massres.yaml
python pm2nbody/train_subregion_force.py  --config configs/subregion_force.yaml
```

### Archivos
| Archivo | Función |
|---------|---------|
| `generate_data_subregion.py` | Genera pares LR/HR + guarda `delta_f_m{hr}_s{n}.npy` pre-computado |
| `train_subregion_force.py` | Entrenamiento patch-based |
| `configs/subregion_force.yaml` | `patch_n: 32`, `n_shell: 0` por defecto |
| `notebooks/subregion_validation.ipynb` | 7 secciones: datos → CNN en HR → MLP → pipeline completo → generalización → localidad |

---

## Pipeline 2 — Generalización de sub-región (`train_subregion_forceres.py`)

### Concepto (más simple)
Una sola simulación, mismas N³ partículas.
- Se calcula la fuerza "exacta" (malla fina) en una **sub-región espacial fija**
- El modelo aprende de esas fuerzas
- Se evalúa qué tan bien predice las fuerzas en **el resto de la simulación**

```
ΔF = F_fine − F_coarse    (corrección resolución de fuerza, dentro de una sola sim)
Train:  parche Lagrangiano fijo  (p.ej. 32³/64³ = 12.5% de la sim)
Test:   todas las demás partículas
```

### Métrica clave
```
Generalisation gap = train_R − test_R
```
- `< 0.05` → excelente, el modelo generaliza bien
- `0.05–0.15` → aceptable
- `> 0.15` → el modelo se sobre-ajusta a la sub-región

### Reutilización de código
```python
# Directamente de train_lag_force.py:
compute_force_pair(pos, mesh_lr, mesh_hr, pos_hr_t=None)
# → (f_coarse, f_fine, delta_f)   mismas partículas, dos mallas
```

### Nueva función de split
```python
# en train_subregion_forceres.py
make_patch_split(n_part, train_patch_n, seed)
# → (train_idx, test_idx, patch_n)
# train_idx: cubo Lagrangiano contiguo de tamaño train_patch_n³
# test_idx:  resto de la simulación
```

### Training loop (diferencia principal)
```python
# Se calculan features y fuerzas para TODAS las partículas
# Pero el train step solo usa train_idx:
params, opt_state, loss = train_step(
    params, opt_state,
    feats_all[train_idx], vel[train_idx], a, delta_f[train_idx], weights[train_idx]
)

# En cada log step, métricas separadas:
train_metrics = eval_region(model, params, feats_all, ..., train_idx, "train/")
test_metrics  = eval_region(model, params, feats_all, ..., test_idx,  "test/")
# → wandb reporta train_R y test_R en cada paso
```

### Workflow
```bash
python pm2nbody/generate_data_single.py --mode test
python pm2nbody/train_subregion_forceres.py --config configs/subregion_forceres.yaml
```

### Archivos
| Archivo | Función |
|---------|---------|
| `generate_data_single.py` | Una sola sim, N³ partículas, sin fuerzas pre-computadas |
| `train_subregion_forceres.py` | Split fijo + log de gap de generalización |
| `configs/subregion_forceres.yaml` | `train_patch_n: 32`, `mesh_lr: 32`, `mesh_hr: 64` |
| `notebooks/subregion_forceres_validation.ipynb` | 6 secciones: sanity → split → correlación features → scatter train/test → mapas espaciales → snapshots |

---

## Comparación de los dos pipelines nuevos

| | Sub-región zoom | Generalización sub-región |
|--|----------------|--------------------------|
| **Simulaciones** | Par LR/HR | Una sola sim |
| **Target** | `F_HR − F_LR@HR` en partículas HR | `F_fine − F_coarse` en mismas partículas |
| **Sub-región** | Parche aleatorio en cada paso | Parche **fijo** (train vs test) |
| **Pregunta** | ¿Puede CNN+MLP predecir fuerzas HR localmente? | ¿Generaliza de una región a toda la sim? |
| **Métrica principal** | MSE improvement vs F_HR | `train_R − test_R` (gap) |
| **Datos pre-computados** | Sí (`delta_f_m{hr}_s{n}.npy`) | No (on-the-fly) |

---

## Corrección de bug en `massres_validation.ipynb`

El notebook no contabilizaba la contribución CNN en modo dos-etapas.

**Error:** `f_corrected = F_LR + ΔF_MLP`  
**Correcto:** `f_corrected = F_LR + ΔF_CNN + ΔF_MLP`

### Celdas modificadas
| Celda | Cambio |
|-------|--------|
| Nueva celda `39b2b4ae` (antes de cell-13) | Carga CNN si existe `cnn_checkpoint`, computa `cnn_delta_f_v_np`, define `IS_TWO_STAGE` y `df_v_target_np` |
| `cell-13` (scatter MLP) | Usa `df_v_target_np` = residual correcto como target; añade 2° scatter total (CNN+MLP) vs ΔF_mass |
| `cell-15b` (mapas de fuerza) | `f_corrected = F_LR + cnn_delta_f_v_np + pred_v_np` |
| `cell-1b27970f` (tabla global) | `snapshot_improvement()` recalcula `vcnn_np` por snapshot |

---

## Secciones añadidas al notebook `massres_validation.ipynb`

### Section 5b — Global force comparison & improvement metrics

| Celda | Visualización |
|-------|--------------|
| 5b-1 | Histogramas superpuestos `|F_LR|`, `|F_corrected|`, `|F_HR|` (lineal + log) |
| 5b-2 | Scatter per-partícula: error antes vs después; histograma de mejora relativa |
| 5b-3 | CDF de `|F − F_HR|` para LR y corregido; split SC vs no-SC |
| 5b-4 | Tabla completa por snapshot: MSE, MAE, P50, P95, frac_improved, R |

---

## Estructura final de archivos nuevos

```
JaxPM/
├── pm2nbody/
│   ├── generate_data_single.py          ← nueva
│   ├── generate_data_subregion.py       ← nueva
│   ├── train_subregion_force.py         ← nueva (zoom sub-región)
│   ├── train_subregion_forceres.py      ← nueva (generalización)
│   └── notebooks/
│       ├── massres_validation.ipynb     ← modificada (Section 5b + bug CNN)
│       ├── subregion_validation.ipynb   ← nueva
│       └── subregion_forceres_validation.ipynb  ← nueva
└── configs/
    ├── subregion_force.yaml             ← nueva
    └── subregion_forceres.yaml          ← nueva
```

`train_lag_massres.py` también fue modificado: se añadió `compute_subregion_force_pair()`.
