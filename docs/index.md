# PM2NBody / Nbodyify — Project Flow & Normalization Conventions

This page documents how the PM2NBody (a.k.a. “Nbodyify”) training pipeline works end‑to‑end, with a focus on **who calls what** and **which quantity is normalized to which unit system**.  
It is intended as living documentation: you can extend and refine sections as the code evolves.

---

## Project goal

The project trains a **correction model** (e.g., a CNN on LR grids or a k-space correction module) to improve a **Particle‑Mesh (PM)** particle dynamics simulation. The training compares **predicted trajectories** (from the PM integrator with/without correction) to **HR targets**, using losses such as position MSE and optional statistical losses (density, power spectrum, cross‑correlation).

A consistent and explicit set of **unit conventions** is critical across:

- Dataset loading
- ODE/PM integration (internal state)
- Loss computation
- Visualization/diagnostics (`plot_eval`)

---

## Unit conventions and normalization

### Key variables

- `mesh_lr`: mesh size used by the training dynamics (e.g., 128).
- `mesh_hr`: mesh size associated with the HR dataset (e.g., 256). This may exist as metadata, but is not necessarily the mesh used by the integrator.
- `box_size`: physical box size (e.g., 256.0). Used only if dataset files come in physical units.

### Adopted convention (mesh units)

To avoid huge extra copies and to remain consistent with CIC/FFT operations, the pipeline uses:

- `positions`, `velocities` in **mesh units**:

\[
  \mathbf{x} \in \left(0, n_{\mathrm{mesh}}\right)^3
\]

where typically \( n_{\mathrm{mesh}} = \texttt{mesh_{lr}} \).

- **Periodicity** enforced with a wrap:

\[
  \mathbf{x} \leftarrow \mathbf{x} \bmod n_{\mathrm{mesh}}
\]

#### Periodic distance (minimal image convention)

To compare two periodic positions \(\mathbf{x}, \mathbf{y}\):

\[
\Delta = \mathbf{x} - \mathbf{y}, \qquad
\Delta \leftarrow \Delta - n_{\mathrm{mesh}}\,\mathrm{round}\!\left(\frac{\Delta}{n_{\mathrm{mesh}}}\right),
\]

so that each component of \(\Delta\) lies in \([ -\tfrac{n_{\mathrm{mesh}}}{2},\tfrac{n_{\mathrm{mesh}}}{2})\).

#### Resolution‑invariant loss scaling

To keep the position MSE scale comparable across different meshes:

\[
\mathrm{MSE}_{\mathrm{pos,norm}} \;=\; \frac{1}{n_{\mathrm{mesh}}^2}\;\mathbb{E}\left[\|\Delta\|_2^2\right].
\]

---

## Components and responsibilities

### Dataset loading

Typical files per simulation (example naming):

- `pos_m{n_mesh}_s{idx}.npy` : particle positions per snapshot
- `vel_m{n_mesh}_s{idx}.npy` : particle velocities per snapshot
- `pot_m{n_mesh}_s{idx}.npy` : particle potential per snapshot
- `pot_grid_m{n_mesh}_s{idx}.npy` : LR potential grid (optional)
- `dens_grid_m{n_mesh}_s{idx}.npy` : LR density grid (optional)

#### `get_data(...)`

Responsibilities:

- Load `.npy` arrays on CPU.
- Slice snapshots if `snapshots` is provided; if `None`, use **all** snapshots.
- Convert to `jax.numpy` (prefer `float32`).
- Ensure the chosen unit convention (here: mesh units).

#### `ResolutionData`

A container that groups tensors for a given resolution:

- `mesh` (int): mesh size used by downstream code (in practice, training uses `mesh_lr`)
- `positions`: `(T, Np, 3)`
- `velocities`: `(T, Np, 3)`
- `potential`: `(T, ...)`
- `potential_grid`, `density_grid`: optional
- `to_device(device)`: **transport only** (device_put only; no rescaling)

#### `PMDataset`

- Holds lists `hr` and `lr` of `ResolutionData`.
- Yields items `{"hr": ..., "lr": ...}`.
- `move_to_device(batch, device)` returns **new** objects on device and **does not mutate** the stored dataset (prevents VRAM growth across steps).

---

## Correction model and dynamics

### `build_network(config.correction_model)`

Builds the correction model depending on `type`, e.g.:

- `cnn`: CNN over LR grids (potential/density)
- `kcorr`: k‑space correction (filter or delta in Fourier domain)

The model is used inside the dynamics.

### `make_ode_fn(mesh_shape, add_correction, model)`

Creates the ODE RHS function for PM integration.

Typical internal steps:

- deposit density on a mesh (CIC)
- solve for potential (FFT)
- if correction enabled: compute correction via the model and apply to potential/force
- update particle state according to the integrator scheme

### `odeint(...)`

Integrates from an initial state across `scale_factors`:

- outputs `pos_pm, vel_pm` shaped `(T, Np, 3)`.

---

## Losses

### `build_loss_fn(training_config, ...)`

Constructs `loss_fn(params, dataset, scale_factors)` based on `training_config.loss`.

#### `mse_positions`

- Calls `get_position_loss(...)`.
- Feeds `pos_lr, vel_lr, pos_hr, vel_hr` and `scale_factors`.
- `get_position_loss`:
  - integrates using `odeint(make_ode_fn(...))`
  - wraps periodicity and computes minimal‑image displacement
  - computes normalized position MSE
  - optional additional terms:
    - `lambda_velocity`
    - `lambda_density`
    - `lambda_cross_corr`
    - `lambda_pk`
- returns `(loss, aux)` where `aux = pos_pm` (mesh units), used by `plot_eval`.

---

## Training

### `train(config, data_dir, output_dir)`

High‑level steps:

1. Build model: `build_network`
2. Build dataloaders: `build_dataloader`
3. Initialize parameters: `initialize_network`
4. Build loss: `build_loss_fn`
5. Build schedule & optimizer: `build_schedule`, `build_optimizer`
6. Training loop:
   - get batch: `next(train_data.iterator)`
   - move to device: `train_data.move_to_device(batch, device)`
   - `update_step` (JIT):
     - `value_and_grad(train_loss_fn)`
     - `optimizer.update`
     - `optax.apply_updates`
7. Validation:
   - compute `loss_fn` on validation set
   - take `aux_first_batch` for `plot_eval`
   - early stopping, schedule stepping, wandb logging

---

## Visualization and diagnostics (`plot_eval`)

`plot_eval` builds density contrast `delta` (via CIC deposit through `get_delta`) and plots:

- 2D projections (slabs) of density/overdensity
- power spectrum ratio \(P(k)/P_{\mathrm{HR}}(k)\)

### Unit requirements for plotting

To avoid unit mismatches:

- `val_pos_pm` (aux) is in **mesh units**.
- `val_data["lr"].positions` and `val_data["hr"].positions` must be interpreted consistently (ideally also in mesh units).
- the deposit mesh used for plotting should be `mesh_plot = val_data["lr"].mesh`.

Also note: projecting **δ** over the whole box tends to cancel structure (δ has mean ≈ 0). Prefer a small slab thickness or plot `ρ = 1 + δ`.

---

## Mermaid diagrams

``` mermaid
sequenceDiagram
  autonumber
  Alice->>John: Hello John, how are you?
  loop Healthcheck
      John->>John: Fight against hypochondria
  end
  Note right of John: Rational thoughts!
  John-->>Alice: Great!
  John->>Bob: How about you?
  Bob-->>John: Jolly good!
```
### High‑level call graph

``` mermaid
flowchart TD
  A["train"] --> B["build_network"]
  A --> C["build_dataloader"]
  C --> C1["load_datasets"]
  C1 --> C2["get_data load_npy select_snapshots"]
  C1 --> C3["ResolutionData HR_LR"]
  C --> D["initialize_network"]

  A --> E["build_loss_fn"]
  E --> E1["get_position_loss get_potential_loss others"]

  A --> F["build_schedule"]
  A --> G["build_optimizer"]

  A --> H["training_loop"]
  H --> H1["next_train_batch"]
  H1 --> H2["move_to_device"]
  H2 --> H3["update_step_jit"]
  H3 --> H31["value_and_grad"]
  H31 --> H32["loss_fn"]
  H32 --> H33["odeint make_ode_fn"]
  H33 --> H34["compute_loss_and_aux"]
  H3 --> H35["optimizer_update_apply"]

  H --> V["validation_loop"]
  V --> V1["loss_fn_validation"]
  V1 --> V2["plot_eval_from_aux"]
```

### Unit flow for `mse_positions`

``` mermaid
flowchart LR
  %% Define styles first (more robust in MkDocs/Mermaid plugins)
  classDef m fill:#efe,stroke:#474,stroke-width:1px;

  subgraph DS["Dataset"]
    D1["pos_lr, vel_lr"]:::m
    D2["pos_hr, vel_hr"]:::m
  end

  subgraph DY["Dynamics"]
    O1["odeint(make_ode_fn)"]:::m
    O2["pos_pm, vel_pm"]:::m
  end

  subgraph LS["Loss"]
    L1["wrap: mod n_mesh"]:::m
    L2["minimal-image distance"]:::m
    L3["MSE / n_mesh^2"]:::m
  end

  D1 --> O1 --> O2 --> L1 --> L2 --> L3
  D2 --> L1
```

Legend: green blocks correspond to the mesh‑unit convention.

---

## Consistency checklist (quick debugging)

### Signs you are in mesh units

- `max(pos_lr[t0]) ~ n_mesh` and `mean ~ n_mesh/2`
- `mean |mod(pos_pm,n_mesh) - pos_pm| ~ 0`

### Common ways scale breaks

1. A `to_device()` that rescales (should be transport‑only).
2. `build_loss_fn` multiplying by `mesh` when the dataset is already in mesh units.
3. `plot_eval` assuming box units and multiplying again by `mesh_plot` (double scaling).
4. Using `mod(1.0)` on mesh‑unit coordinates.

---

## Notes for future extensions

- Grid‑based losses (density, pk, cross‑corr) must use the **same mesh** as the deposit/FFT.
- For huge datasets:
  - keep `float32`
  - avoid repeated rescaling
  - do not mutate dataset objects when moving to device
  - validate with small subsets for fast diagnostics