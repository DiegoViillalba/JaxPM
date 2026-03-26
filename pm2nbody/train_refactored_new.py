# ==============================================================================
# train_refactored.py
# Memory-safe training loop for JaxPM neural correction models.
#
# Memory fixes applied vs previous version:
#   1. XLA_PYTHON_CLIENT_PREALLOCATE set exactly once, before any JAX import
#   2. val loop: jax.device_get() + explicit del after each batch to free device
#      buffers immediately instead of accumulating on GPU
#   3. val loop: loss_fn wrapped in jax.jit to avoid retracing & alloc spikes
#   4. plot_eval called AFTER all val batches finish + device buffers freed
#   5. best_params stored as CPU numpy dict (hk.data_structures.to_mutable_dict
#      + jax.device_get), not live device arrays — avoids doubling param memory
#   6. Matplotlib figures explicitly closed after every wandb.log call
#   7. Removed duplicate os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# ==============================================================================

import os

# ── Must be set before importing JAX ─────────────────────────────────────────
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.85"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

import sys
import yaml
import pickle
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
from absl import flags

import jax
import jax.numpy as jnp
from jax import config as jax_config
import optax
import haiku as hk
from flax.training.early_stopping import EarlyStopping
import wandb

import jax_cosmo as jc
from jaxpm.nn import CNN, NeuralSplineFourierFilter
from jaxpm.painting import compensate_cic
from jaxpm.utils import power_spectrum
from jaxpm.pm import get_delta

from read_data import load_datasets
from loss import (
    get_frozen_potential_loss,
    get_potential_loss,
    get_position_loss,
)

###### WST Integration #####
from wst import WaveletScatteringTransform

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── JAX / Matplotlib global config ───────────────────────────────────────────
jax_config.update("jax_enable_x64", False)

mpl.rcParams.update({"text.usetex": False, "font.family": "serif"})
plt.style.use("default")

# ── Default paths ─────────────────────────────────────────────────────────────
DEFAULT_DATA_DIR  = Path("/cosmos_storage/home/diegovillalba/JaxPM/data/")
DEFAULT_MODEL_DIR = Path("/cosmos_storage/home/diegovillalba/JaxPM/models/")

flags.DEFINE_string("config", "", "Path to a YAML config file (optional).")


# ==============================================================================
# 1. Configuration
# ==============================================================================

class AttrDict(dict):
    """Dict with attribute access and recursive to_dict()."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def __setattr__(self, k, v):
        self[k] = v

    def to_dict(self):
        def conv(x):
            if isinstance(x, (AttrDict, dict)):
                return {k: conv(v) for k, v in x.items()}
            return x
        return conv(self)


def to_attrdict(d: Dict[str, Any]) -> AttrDict:
    out = AttrDict()
    for k, v in d.items():
        out[k] = to_attrdict(v) if isinstance(v, dict) else v
    return out


def deep_update(base: dict, upd: dict) -> dict:
    """Recursive merge: upd overwrites base."""
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def default_config() -> AttrDict:
    cfg = {
        "data": {
            "mesh_lr": 128,
            "mesh_hr": 256,
            "n_train_sims": 1,
            "n_val_sims": 1,
            "n_test_sims": 1,
            "snapshots": None,
            "box_size": 256.0,
            "n_snapshots": 50,
            "n_particles": 128,
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
            "add_particle_velocities": True,
            "n_knots": 16,
            "latent_size": 64,
        },
        "training": {
            "seed": 0,
            "n_steps": 150,
            "batch_size": 1,
            "patience": 20,
            "checkpoint_every": 5,
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
                "n_steps": 100,
                "factor": 0.5,
                "patience": 5,
                "min_lr": 5e-5,
            },
        },
        "wandb": {"project": "pm2nbody"},
    }
    return to_attrdict(cfg)


def load_config_yaml(path: str) -> AttrDict:
    cfg = default_config().to_dict()
    with open(path, "r") as f:
        user_cfg = yaml.safe_load(f) or {}
    cfg = deep_update(cfg, user_cfg)
    return to_attrdict(cfg)


# ==============================================================================
# 2. Model construction
# ==============================================================================

def build_network(config):
    logger.info(f"Building correction model: {config.type}")

    def CNNCorr(x, positions, scale_factors, velocities):
        cnn = CNN(
            channels_hidden_dim=config.channels_hidden_dim,
            n_convolutions=config.n_convolutions,
            n_fully_connected=config.n_fully_connected,
            input_dim=config.input_dim,
            output_dim=3 if config.type == "cnn_force" else 1,
            kernel_size=config.kernel_size,
            pad_periodic=config.pad_periodic,
            embed_globals=config.embed_globals,
            n_globals_embedding=config.n_globals_embedding,
            globals_embedding_dim=config.globals_embedding_dim,
            global_conditioning=config.global_conditioning,
            use_attention_interpolation=config.use_attention_interpolation,
            add_particle_velocities=config.add_particle_velocities,
        )
        return cnn(
            x=x,
            positions=positions,
            global_features=scale_factors,
            return_features=False,
            velocities=velocities,
        )

    def KCorr(x, scale_factors):
        return NeuralSplineFourierFilter(
            n_knots=config.n_knots,
            latent_size=config.latent_size,
        )(x, scale_factors)

    if config.type in ("cnn", "cnn_force"):
        return hk.without_apply_rng(hk.transform(CNNCorr))
    elif config.type == "kcorr":
        return hk.without_apply_rng(hk.transform(KCorr))
    elif config.type == "cnn+kcorr":
        return {
            "cnn":   hk.without_apply_rng(hk.transform(CNNCorr)),
            "kcorr": hk.without_apply_rng(hk.transform(KCorr)),
        }
    elif config.type == "cnn_wst":
        logger.info("Using CNN + WST correction model.")

        def CNNWSTCorr(x, positions, scale_factors, velocities):

            wst   = WaveletScatteringTransform(
                J=config.wst_J, L=config.wst_L, normalize=True, name="wst"
            )
            x_wst = wst(x)   # [N,N,N, C + J*L]

            # input_dim dinámico — soporta C=1 o C=2 según need_grid
            actual_input_dim = x.shape[-1] + config.wst_J * config.wst_L

            cnn = CNN(
                channels_hidden_dim=config.channels_hidden_dim,
                n_convolutions=config.n_convolutions,
                n_fully_connected=config.n_fully_connected,
                input_dim=actual_input_dim,   # ← dinámico, no hardcodeado
                output_dim=1,
                kernel_size=config.kernel_size,
                pad_periodic=config.pad_periodic,
                embed_globals=config.embed_globals,
                n_globals_embedding=config.n_globals_embedding,
                globals_embedding_dim=config.globals_embedding_dim,
                global_conditioning=config.global_conditioning,
                use_attention_interpolation=config.use_attention_interpolation,
                add_particle_velocities=config.add_particle_velocities,
            )
            return cnn(
                x=x_wst,
                positions=positions,
                global_features=scale_factors,
                return_features=False,
                velocities=velocities,
            )

        return hk.without_apply_rng(hk.transform(CNNWSTCorr))

    else:
        raise NotImplementedError(f"Unknown correction model type: {config.type}")


def initialize_network(data_sample, neural_net, seed: int = 42, model_type: str = "cnn"):
    logger.info(f"Initializing network (seed={seed}, type={model_type})")
    rng           = jax.random.PRNGKey(seed)
    grid_full     = data_sample["lr"].get_grid()
    grid_init     = grid_full[0]
    pos_init      = data_sample["lr"].positions[0]
    vel_init      = data_sample["lr"].velocities[0]
    scale_init    = jnp.array(1.0)

    if model_type == "kcorr":
        return neural_net.init(rng, grid_init, scale_init)
    elif model_type in ("cnn", "cnn_force"):
        return neural_net.init(rng, grid_init, pos_init, scale_init, vel_init)
    elif model_type == "cnn+kcorr":
        return {
            "kcorr": neural_net["kcorr"].init(rng, grid_init, scale_init),
            "cnn":   neural_net["cnn"].init(rng, grid_init, pos_init, scale_init, None),
        }
    elif model_type in ("cnn", "cnn_force", "cnn_wst"):
      return neural_net.init(rng, grid_init, pos_init, scale_init, vel_init)
    else:
        raise NotImplementedError(f"Unknown model type: {model_type}")


# ==============================================================================
# 3. Loss function builder
# ==============================================================================

def build_loss_fn(training_config, neural_net, cosmology, correction_type, mesh_lr: int):
    
    logger.info(f"Building loss: {training_config.loss} | correction: {correction_type}")

    correction_type = "cnn"


    if training_config.loss == "mse_frozen_potential":
        single_loss_fn = get_frozen_potential_loss(neural_net=neural_net)
        vmap_loss      = jax.vmap(single_loss_fn, in_axes=(None, 0, 0, 0, 0, 0))

        def loss_fn(params, dataset, scale_factors, grid=None):
            loss_array = vmap_loss(
                params,
                dataset["lr"].grid,
                dataset["lr"].positions * dataset["lr"].mesh,
                dataset["lr"].potential,
                dataset["hr"].potential,
                scale_factors,
            )
            T    = loss_array.shape[0]
            mask = jnp.arange(T) <= training_config.max_idx
            return jnp.sum(loss_array * mask) / jnp.clip(jnp.sum(mask), a_min=1.0), None

    elif training_config.loss == "mse_potential":
        single_loss_fn = get_potential_loss(
            neural_net=neural_net,
            cosmology=cosmology,
            correction_type=correction_type,
        )

        def loss_fn(params, dataset, scale_factors, grid=None):
            T = min(
                scale_factors.shape[0],
                dataset["lr"].positions.shape[0],
                dataset["hr"].positions.shape[0],
            )
            return single_loss_fn(
                params,
                grid,
                dataset["lr"].positions[:T],
                dataset["lr"].velocities[:T],
                dataset["hr"].potential[:T],
                scale_factors[:T],
            )

    elif training_config.loss == "mse_positions":
        single_loss_fn = get_position_loss(
            neural_net=neural_net,
            cosmology=cosmology,
            correction_type=correction_type,
            weight_snapshots=training_config.weight_snapshots,
            n_mesh=mesh_lr,
            lambda_pos=training_config.lambda_pos,
            lambda_velocity=training_config.lambda_velocity,
            lambda_density=training_config.lambda_density,
            lambda_pk=training_config.lambda_pk,
            lambda_cross_corr=training_config.lambda_cross_corr,
            log_pos=training_config.log_pos,
            fractional_mse=training_config.fractional_mse,
        )

        def loss_fn(params, dataset, scale_factors, grid=None):
            T = min(
                scale_factors.shape[0],
                dataset["lr"].positions.shape[0],
                dataset["hr"].positions.shape[0],
            )
            return single_loss_fn(
                params,
                dataset["lr"].positions[:T],
                dataset["lr"].velocities[:T],
                dataset["hr"].positions[:T],
                dataset["hr"].velocities[:T],
                scale_factors[:T],
            )

    else:
        raise ValueError(f"Unknown loss type: {training_config.loss}")

    return loss_fn


# ==============================================================================
# 4. Data, optimizer, scheduler
# ==============================================================================

def build_dataloader(config, data_dir=DEFAULT_DATA_DIR, need_grid=False):
    cosmology  = jc.Planck15(Omega_c=0.25, sigma8=0.8)
    data_path  = (
        data_dir
        / f"matched_{config.mesh_lr}_{config.mesh_hr}"
          f"_L{config.box_size:.1f}_S{config.n_snapshots}_Np{config.n_particles}/"
    )
    scale_factors = jnp.load(data_path / "scale_factors.npy")

    snapshots = None
    if config.snapshots is not None:
        snapshots     = jnp.array(config.snapshots)
        scale_factors = scale_factors[snapshots]

    train_data, val_data, test_data = load_datasets(
        config.n_train_sims,
        config.n_val_sims,
        config.n_test_sims,
        mesh_hr=config.mesh_hr,
        mesh_lr=config.mesh_lr,
        data_dir=data_path,
        snapshots=snapshots,
        box_size=config.box_size,
        need_grid=need_grid,
    )
    logger.info(
        f"Datasets — Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}"
    )
    return cosmology, scale_factors, train_data, val_data, test_data


def build_schedule(config):
    if config.type == "cosine":
        return optax.warmup_cosine_decay_schedule(
            init_value=config.initial_lr,
            peak_value=config.peak_value,
            warmup_steps=config.warmup_steps,
            decay_steps=config.n_steps,
        )
    raise NotImplementedError(f"Unknown schedule type: {config.type}")


def build_optimizer(config, params, schedule):
    optimizer = optax.MultiSteps(
        optax.chain(
            optax.clip(1.0),
            optax.inject_hyperparams(optax.adamw)(
                learning_rate=schedule,
                weight_decay=config.weight_decay,
            ),
        ),
        every_k_schedule=config.batch_size,
        use_grad_mean=True,
    )
    return optimizer, optimizer.init(params)


# ==============================================================================
# 5. Checkpointing
# ==============================================================================

def checkpoint(run_dir, loss, params, prefix, step=None):
    """
    Save params to disk as CPU numpy.
    params may be live device arrays — we pull them to host before pickling
    to avoid accumulating GPU allocations from multiple checkpoint copies.
    """
    filename = f"{prefix}_{loss:.3f}_weights"
    filename += f"_{step}.pkl" if step is not None else ".pkl"

    # Pull to CPU before serialising — avoids keeping a second GPU copy
    cpu_params = jax.device_get(params)
    with open(run_dir / filename, "wb") as f:
        pickle.dump(hk.data_structures.to_immutable_dict(cpu_params), f)

    logger.info(f"Checkpoint saved: {filename}")


# ==============================================================================
# 6. Evaluation helpers
# ==============================================================================

def print_initial_lr_loss(val_data):
    """Compute naive LR-vs-HR baselines before any training."""
    pos_losses, vel_losses, pot_losses = [], [], []

    for batch in val_data:
        lr_pos = jnp.mod(batch["lr"].positions, 1.0)
        hr_pos = jnp.mod(batch["hr"].positions, 1.0)
        d      = lr_pos - hr_pos
        d      = d - jnp.round(d)
        pos_losses.append(float(jnp.mean(jnp.sum(d * d, axis=-1))))

        dv = batch["lr"].velocities - batch["hr"].velocities
        vel_losses.append(float(jnp.mean(jnp.sum(dv * dv, axis=-1))))

        dp = batch["lr"].potential - batch["hr"].potential
        pot_losses.append(float(jnp.mean(dp * dp)))

    logger.info(f"Baseline Pos MSE  = {np.mean(pos_losses):.5f}")
    logger.info(f"Baseline Vel MSE  = {np.mean(vel_losses):.5f}")
    logger.info(f"Baseline Pot MSE  = {np.mean(pot_losses):.5f}")


def plot_eval(
    val_pos_pm,
    val_data,
    max_idx=None,
    box_size=256.0,
    fig_label="val",
    use_wandb=True,
    plot_log=True,
    slab_thickness=5,
):
    """
    Two-panel eval plot:
      1. 2D slab projection of log10(1+delta)
      2. P(k)/P_HR(k) ratio
    All arrays pulled to CPU numpy before any computation.
    Figures explicitly closed after logging to free matplotlib memory.
    """
    max_idx   = max_idx if max_idx is not None else -1
    mesh_plot = int(val_data["lr"].mesh)

    def to_mesh(pos):
        pos = np.asarray(pos)
        return pos * mesh_plot if np.max(pos) <= 2.0 else pos

    pos_pm = np.mod(to_mesh(val_pos_pm[max_idx]),               mesh_plot)
    pos_hr = np.mod(to_mesh(val_data["hr"].positions[max_idx]), mesh_plot)
    pos_lr = np.mod(to_mesh(val_data["lr"].positions[max_idx]), mesh_plot)

    delta_pm = np.asarray(get_delta(pos_pm, (mesh_plot,) * 3))
    delta_hr = np.asarray(get_delta(pos_hr, (mesh_plot,) * 3))
    delta_lr = np.asarray(get_delta(pos_lr, (mesh_plot,) * 3))

    z0  = mesh_plot // 2
    h   = max(1, slab_thickness // 2)
    sl  = slice(max(z0 - h, 0), min(z0 + h, mesh_plot))

    projs = {}
    for name, d in (("LR", delta_lr), ("PM", delta_pm), ("HR", delta_hr)):
        p = d[:, :, sl].sum(axis=-1)
        if plot_log:
            p = np.log10(np.clip(1.0 + p, 1e-4, None))
        projs[name] = p

    suffix = " (log)" if plot_log else ""
    fig_delta, axes = plt.subplots(ncols=3, figsize=(12, 5))
    cmap = "cividis"
    axes[0].imshow(projs["LR"], cmap=cmap); axes[0].set_title(f"LR{suffix}", fontsize=20)
    axes[1].imshow(projs["PM"], cmap=cmap); axes[1].set_title(f"PM+corr{suffix}", fontsize=20)
    axes[2].imshow(projs["HR"], cmap=cmap); axes[2].set_title(f"HR{suffix}", fontsize=20)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()

    if use_wandb:
        wandb.log({f"{fig_label}_delta": wandb.Image(fig_delta)})
    else:
        plt.show()
    plt.close(fig_delta)          # FIX: always close — avoids matplotlib memory leak

    # ── Power spectrum ──────────────────────────────────────────────────────
    def get_pk(delta):
        d = compensate_cic(np.asarray(delta))
        k, pk = power_spectrum(
            d,
            boxsize=np.array([box_size] * 3),
            kmin=np.pi / box_size,
            dk=2 * np.pi / box_size,
        )
        return np.asarray(k), np.asarray(pk)

    k, pk_hr = get_pk(delta_hr)
    _, pk_lr  = get_pk(delta_lr)
    _, pk_pm  = get_pk(delta_pm)

    valid = (
        np.isfinite(k) & np.isfinite(pk_hr) & (pk_hr > 0)
        & np.isfinite(pk_lr) & np.isfinite(pk_pm)
    )
    k, pk_hr, pk_lr, pk_pm = k[valid], pk_hr[valid], pk_lr[valid], pk_pm[valid]

    fig_pk, ax_pk = plt.subplots(figsize=(8, 6))
    ax_pk.axhline(1, linestyle="dashed", color="black")
    if k.size > 0:
        ax_pk.semilogx(k, pk_lr / pk_hr, label="LR")
        ax_pk.semilogx(k, pk_pm / pk_hr, label="PM+corr")
    else:
        ax_pk.text(0.5, 0.5, "No valid P(k) bins.", ha="center", va="center",
                   transform=ax_pk.transAxes)
    ax_pk.legend()
    ax_pk.set_xlabel(r"$k$ [$h\ \mathrm{Mpc}^{-1}$]")
    ax_pk.set_ylabel(r"$P(k)/P_{\rm HR}(k)$")
    ax_pk.set_title("Power spectrum ratio")

    if use_wandb:
        wandb.log({f"{fig_label}_pk": wandb.Image(fig_pk)})
    else:
        plt.show()
    plt.close(fig_pk)             # FIX: always close


# ==============================================================================
# 7. Main training loop
# ==============================================================================

def train(config=None, data_dir=DEFAULT_DATA_DIR, output_dir=DEFAULT_MODEL_DIR):

    # ── Setup ────────────────────────────────────────────────────────────────
    need_grid = config.training.loss in ("mse_potential", "mse_frozen_potential")

    neural_net = build_network(config.correction_model)

    cosmology, scale_factors, train_data, val_data, test_data = build_dataloader(
        config.data, data_dir=data_dir, need_grid=need_grid
    )
    logger.info(
        f"Dataset — Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}"
    )

    params = initialize_network(
        train_data[0],
        neural_net=neural_net,
        model_type=config.correction_model.type,
    )

    # ── WandB ────────────────────────────────────────────────────────────────
    run     = wandb.init(project=config.wandb.project, config=config.to_dict(), dir=output_dir)
    run_dir = output_dir / run.name
    run_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"Run: {run.name}  →  {run_dir}")
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(config.to_dict(), f)

    # ── Loss / optimizer ─────────────────────────────────────────────────────
    loss_fn = build_loss_fn(
        config.training,
        neural_net,
        cosmology,
        correction_type=config.correction_model.type,
        mesh_lr=int(train_data[0]["lr"].mesh),
    )
    schedule           = build_schedule(config.training.schedule)
    optimizer, opt_state = build_optimizer(config.training, params=params, schedule=schedule)

    print_initial_lr_loss(val_data)

    # ── FIX: jit the val loss separately so it doesn't retrace each eval step,
    #   which would cause a fresh XLA compilation + temporary allocation spike.
    @jax.jit
    def val_loss_step(params, batch, scale_factors, grid):
        out = loss_fn(params=params, dataset=batch, scale_factors=scale_factors, grid=grid)
        return out if isinstance(out, tuple) else (out, None)

    # ── Train step ───────────────────────────────────────────────────────────
    def train_loss_fn(p, batch, sf, grid):
        out = loss_fn(params=p, dataset=batch, scale_factors=sf, grid=grid)
        return out[0] if isinstance(out, tuple) else out

    @jax.jit
    def update_step(params, opt_state, batch, scale_factors, grid):
        loss, grads  = jax.value_and_grad(train_loss_fn)(params, batch, scale_factors, grid)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params       = optax.apply_updates(params, updates)
        return loss, params, opt_state

    # ── Best-params tracking on CPU ───────────────────────────────────────────
    # FIX: store best_params as a CPU dict, not live device arrays.
    # Keeping a second copy of params on-device doubles GPU param memory.
    best_params_cpu = jax.device_get(params)
    best_loss       = float("inf")
    early_stop      = EarlyStopping(patience=config.training.patience)

    eval_freq = 10 * config.training.batch_size
    pbar      = tqdm(range(config.training.n_steps), desc="Training")

    for step in pbar:

        # ── Train ─────────────────────────────────────────────────────────
        batch = next(train_data.iterator)
        batch = train_data.move_to_device(batch, device=jax.devices()[0], build_grid=need_grid)
        grid  = batch["lr"].grid if need_grid else None

        train_loss, params, opt_state = update_step(
            params, opt_state, batch, scale_factors, grid
        )
        pbar.set_postfix({"loss": float(jax.device_get(train_loss))})

        # ── Validation ────────────────────────────────────────────────────
        if step > 0 and step % eval_freq == 0:
            val_loss_accum  = 0.0
            val_count       = 0
            aux_for_plot    = None
            batch_for_plot  = None

            for i, val_batch in enumerate(val_data):
                val_batch = val_data.move_to_device(
                    val_batch, device=jax.devices()[0], build_grid=need_grid
                )
                grid_val = val_batch["lr"].grid if need_grid else None

                vl, aux = val_loss_step(params, val_batch, scale_factors, grid_val)

                # FIX: pull scalar to CPU immediately and release device buffer
                val_loss_accum += float(jax.device_get(vl))
                val_count      += 1

                if i == 0:
                    # FIX: materialise aux to CPU right away so the device
                    # buffer for all T snapshots is freed before the next batch
                    aux_for_plot   = jax.device_get(aux) if aux is not None else None
                    batch_for_plot = val_batch

                # FIX: explicitly delete device tensors from this batch
                del vl, aux, val_batch

            val_loss = val_loss_accum / max(val_count, 1)

            # ── Plots (after val loop, on CPU data) ──────────────────────
            if aux_for_plot is not None:
                T            = aux_for_plot.shape[0]
                plot_indices = [0, T // 2, T - 1]
                for idx in plot_indices:
                    plot_eval(
                        aux_for_plot,
                        batch_for_plot,
                        max_idx=idx,
                        fig_label=f"val_t{idx:02d}",
                        slab_thickness=64,
                        use_wandb=True,
                    )
            else:
                logger.info("Val: loss_fn returned no aux — skipping plot_eval.")

            # ── Early stopping & best params ──────────────────────────────
            early_stop = early_stop.update(val_loss)
            if early_stop.has_improved:
                # FIX: copy to CPU dict — not a device reference
                best_params_cpu = jax.device_get(params)
                best_loss       = val_loss

            if hasattr(schedule, "step"):
                schedule.step(val_loss)

            lr = float(opt_state.inner_opt_state[1].hyperparams["learning_rate"])
            wandb.log(
                {
                    "train_loss": float(jax.device_get(train_loss)),
                    "val_loss":   val_loss,
                    "learning_rate": lr,
                },
                step=step,
            )
            pbar.set_postfix({"train": float(jax.device_get(train_loss)), "val": val_loss})

            if early_stop.should_stop:
                logger.info(f"Early stopping at step {step}.")
                break

        # ── Checkpoint ───────────────────────────────────────────────────
        if step > 0 and step % config.training.checkpoint_every == 0:
            checkpoint(run_dir, loss=float(jax.device_get(train_loss)),
                       params=params, prefix="train", step=step)

    # ── Final evaluation ─────────────────────────────────────────────────────
    checkpoint(run_dir, loss=best_loss, params=best_params_cpu, prefix="best")

    test_batch = val_data.move_to_device(test_data[0], device=jax.devices()[0], build_grid=need_grid)
    grid_test  = test_batch["lr"].grid if need_grid else None

    # FIX: restore best params to device only for final eval, then discard
    best_params_device = jax.device_put(best_params_cpu)
    out = loss_fn(best_params_device, test_batch, scale_factors, grid=grid_test)
    del best_params_device

    test_loss, test_aux = out if isinstance(out, tuple) else (out, None)
    logger.info(f"Test loss = {float(jax.device_get(test_loss)):.5f}")

    if test_aux is not None:
        plot_eval(
            jax.device_get(test_aux),
            test_batch,
            max_idx=None,
            fig_label="test",
            use_wandb=True,
        )

    wandb.finish()
    return best_loss


# ==============================================================================
# 8. Entry point
# ==============================================================================

if __name__ == "__main__":
    FLAGS = flags.FLAGS
    FLAGS(sys.argv)

    cfg = load_config_yaml(FLAGS.config) if FLAGS.config else default_config()

    import pprint
    logger.info("Config:")
    pprint.pprint(cfg.to_dict())

    train(cfg)
