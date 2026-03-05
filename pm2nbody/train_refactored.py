import os

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# opcional: limita el pool inicial para evitar acaparar
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.85"
import sys
import yaml
import pickle
from pathlib import Path
from typing import Any, Dict


# Disable XLA preallocation to avoid hogging GPU memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

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
# If using Plateau scheduling in the future, uncomment:
# from jaxpm.nn_utils import ReduceLROnPlateau

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# --- Global Configurations ---
jax_config.update("jax_enable_x64", False)

mpl.rcParams.update(
    {
        "text.usetex": False,
        "font.family": "serif",
    }
)
plt.style.use("default")

DEFAULT_DATA_DIR = Path("/cosmos_storage/home/diegovillalba/JaxPM/data/")
DEFAULT_MODEL_DIR = Path("/cosmos_storage/home/diegovillalba/JaxPM/models/")


# ==========================================
# Configuration Utilities
# ==========================================


class AttrDict(dict):
    """Dictionary with attribute access and recursive to_dict()."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def __setattr__(self, k, v):
        self[k] = v

    def to_dict(self):
        def conv(x):
            if isinstance(x, AttrDict) or isinstance(x, dict):
                return {k: conv(v) for k, v in x.items()}
            return x

        return conv(self)


def to_attrdict(d: Dict[str, Any]) -> AttrDict:
    out = AttrDict()
    for k, v in d.items():
        out[k] = to_attrdict(v) if isinstance(v, dict) else v
    return out


def deep_update(base: dict, upd: dict) -> dict:
    """Recursive merge: 'upd' overwrites 'base'."""
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
            "n_train_sims": 10,
            "n_val_sims": 1,
            "n_test_sims": 1,
            "snapshots": None,
            "box_size": 256.0,
            "n_snapshots": 50,
            "n_particles": 128,
        },
        "correction_model": {
            "type": "cnn",  # Oprtions aviable kcorr,cnn
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
            "n_steps": 100,
            "batch_size": 1,
            "patience": 20,
            "checkpoint_every": 5,
            "sample_snapshots": False,  # Caused JIT compilation issues, so set to False for now
            "loss": "mse_positions",  # options: mse_frozen_potential, mse_potential, mse_positions
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
        "wandb": {
            "project": "pm2nbody",
        },
    }
    return to_attrdict(cfg)


def load_config_yaml(path: str) -> AttrDict:
    cfg = default_config().to_dict()
    with open(path, "r") as f:
        user_cfg = yaml.safe_load(f) or {}
    cfg = deep_update(cfg, user_cfg)
    return to_attrdict(cfg)


flags.DEFINE_string("config", "", "Path to a YAML configuration file (optional).")


# ==========================================
# Model & Loss Definitions
# ==========================================


def build_network(config):
    logger.info(f"Building correction model of type: {config.type}")

    def CNNCorr(x, positions, scale_factors, velocities):
        # logger.info("Initializing CNN correction model with config:")
        # logger.info(f"  channels_hidden_dim: {config.channels_hidden_dim}")
        # logger.info(f"  n_convolutions: {config.n_convolutions}")
        # logger.info(f"  n_fully_connected: {config.n_fully_connected}")
        # logger.info(f"  input_dim: {config.input_dim}")
        # logger.info(f"  kernel_size: {config.kernel_size}")
        # logger.info(f"  pad_periodic: {config.pad_periodic}")
        # logger.info(f"  embed_globals: {config.embed_globals}")
        # logger.info(f"  n_globals_embedding: {config.n_globals_embedding}")
        # logger.info(f"  globals_embedding_dim: {config.globals_embedding_dim}")
        # logger.info(f"  global_conditioning: {config.global_conditioning}")
        # logger.info(f"  use_attention_interpolation: {config.use_attention_interpolation}")
        # logger.info(f"  add_particle_velocities: {config.add_particle_velocities}")
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
            n_knots=config.n_knots, latent_size=config.latent_size
        )(x, scale_factors)

    if config.type in ["cnn", "cnn_force"]:
        logger.info("Using CNN-based correction model.")
        return hk.without_apply_rng(hk.transform(CNNCorr))
    elif config.type == "kcorr":
        logger.info("Using Kernel-based correction model.")
        return hk.without_apply_rng(hk.transform(KCorr))
    elif config.type == "cnn+kcorr":
        logger.info("Using combined CNN + Kernel correction model.")
        return {
            "cnn": hk.without_apply_rng(hk.transform(CNNCorr)),
            "kcorr": hk.without_apply_rng(hk.transform(KCorr)),
        }
    else:
        raise NotImplementedError(
            f"Correction model type {config.type} not implemented"
        )


def initialize_network(
    data_sample, neural_net, seed: int = 42, model_type: str = "cnn"
):
    logger.info(
        f"Initializing network parameters with seed {seed} for model type {model_type}"
    )
    rng = jax.random.PRNGKey(seed)
    # grid_input_init = data_sample["lr"].grid[0]
    grid_full = data_sample["lr"].get_grid()
    grid_input_init = grid_full[0]
    pos_init = data_sample["lr"].positions[0]
    vel_init = data_sample["lr"].velocities[0]
    scale_init = jnp.array(1.0)

    if model_type == "kcorr":
        logger.info("Initializing only the Kernel correction model parameters.")
        return neural_net.init(rng, grid_input_init, scale_init)
    elif model_type in ["cnn", "cnn_force"]:
        logger.info("Initializing only the CNN correction model parameters.")
        return neural_net.init(rng, grid_input_init, pos_init, scale_init, vel_init)
    elif model_type == "cnn+kcorr":
        logger.info("Initializing both CNN and Kernel correction model parameters.")
        return {
            "kcorr": neural_net["kcorr"].init(rng, grid_input_init, scale_init),
            "cnn": neural_net["cnn"].init(
                rng, grid_input_init, pos_init, scale_init, None
            ),
        }


def build_loss_fn(
    training_config, neural_net, cosmology, correction_type, mesh_lr: int
):
    logger.info(
        f"Building loss function for training with loss type: {training_config.loss}"
    )
    logger.info(f"Correction model type for loss function: {correction_type}")

    if training_config.loss == "mse_frozen_potential":
        single_loss_fn = get_frozen_potential_loss(neural_net=neural_net)
        vmap_loss = jax.vmap(single_loss_fn, in_axes=(None, 0, 0, 0, 0, 0))

        def loss_fn(params, dataset, scale_factors, max_idx, grid=None):
            # 1. Compute loss for ALL snapshots (static shapes)
            loss_array = vmap_loss(
                params,
                dataset["lr"].grid,
                dataset["lr"].positions * dataset["lr"].mesh,
                dataset["lr"].potential,
                dataset["hr"].potential,
                scale_factors,
            )

            # 2. Create a boolean mask for valid indices
            # max_idx determines how many snapshots to include
            T = loss_array.shape[0]
            mask = jnp.arange(T) <= max_idx

            # 3. Apply mask and compute mean only over valid snapshots
            masked_loss = jnp.sum(loss_array * mask)
            return masked_loss / jnp.clip(jnp.sum(mask), a_min=1.0)

    elif training_config.loss == "mse_potential":
        single_loss_fn = get_potential_loss(
            neural_net=neural_net,
            cosmology=cosmology,
            correction_type=correction_type,
        )
        logger.info("Using MSE loss on potential (trajectory loss).")

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
                dataset["hr"].potential[:T],  # <- target potencial HR
                scale_factors[:T],
            )

    elif training_config.loss == "mse_positions":
        logger.info("Using MSE loss on positions with vmap over snapshots.")
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

        def loss_fn(params, dataset, scale_factors,grid=None):
            T = min(
                scale_factors.shape[0],
                dataset["lr"].positions.shape[0],
                dataset["hr"].positions.shape[0],
            )
            # m = jnp.asarray(mesh_lr, dtype=dataset["lr"].positions.dtype)

            return single_loss_fn(
                params,
                dataset["lr"].positions[:T],
                dataset["lr"].velocities[:T],
                dataset["hr"].positions[:T],
                dataset["hr"].velocities[:T],
                scale_factors[:T],
            )

        # def loss_fn(params, dataset, scale_factors, max_idx):
        #     return single_loss_fn(
        #         params,
        #         dataset["lr"].positions[:max_idx] * dataset["lr"].mesh,
        #         dataset["lr"].velocities[:max_idx] * dataset["lr"].mesh,
        #         dataset["hr"].positions[:max_idx] * dataset["lr"].mesh,
        #         dataset["hr"].velocities[:max_idx] * dataset["lr"].mesh,
        #         scale_factors[:max_idx],
        #     )
    else:
        raise ValueError(f"Unknown loss type: {training_config.loss}")

    return loss_fn


# ==========================================
# Data & Training Utilities
# ==========================================


def build_dataloader(config, data_dir=DEFAULT_DATA_DIR, need_grid=False):
    cosmology = jc.Planck15(Omega_c=0.25, sigma8=0.8)
    logger.info("Cosmology created")
    data_path = (
        data_dir
        / f"matched_{config.mesh_lr}_{config.mesh_hr}_L{config.box_size:.1f}_S{config.n_snapshots}_Np{config.n_particles}/"
    )
    scale_factors = jnp.load(data_path / "scale_factors.npy")

    if config.snapshots is not None:
        logger.info(f"Using specified snapshots: {config.snapshots}")
        snapshots = jnp.array(config.snapshots)
        scale_factors = scale_factors[snapshots]
    else:
        snapshots = None

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
        f"Datasets Created: Train ({len(train_data)}), Val ({len(val_data)}), Test ({len(test_data)})"
    )
    logger.info(f"Dataset device: {train_data[0]['lr'].density_grid.device}")
    return cosmology, scale_factors, train_data, val_data, test_data


def build_schedule(config):
    if config.type == "cosine":
        return optax.warmup_cosine_decay_schedule(
            init_value=config.initial_lr,
            peak_value=config.peak_value,
            warmup_steps=config.warmup_steps,
            decay_steps=config.n_steps,
        )
    # Require explicit import of ReduceLROnPlateau if needed
    elif config.type == "plateau":
        raise NotImplementedError(
            "Ensure ReduceLROnPlateau is imported and implemented before using."
        )


def build_optimizer(config, params, schedule=None):
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
    opt_state = optimizer.init(params)
    return optimizer, opt_state


def checkpoint(run_dir, loss, params, prefix, step=None):
    filename = f"{prefix}_{loss:.3f}_weights" + (
        f"_{step}.pkl" if step is not None else ".pkl"
    )
    with open(run_dir / filename, "wb") as f:
        state_dict = hk.data_structures.to_immutable_dict(params)
        pickle.dump(state_dict, f)


# ==========================================
# Evaluation & Plotting
# ==========================================


def print_initial_lr_loss(val_data):
    logger.info("Calculating initial loss on validation set with untrained model...")
    val_pos_loss, val_vel_loss, val_pot_loss = [], [], []

    for val_batch in val_data:
        # Convention: positions/velocities are in BOX UNITS [0,1)
        lr_pos = jnp.mod(val_batch["lr"].positions, 1.0)
        hr_pos = jnp.mod(val_batch["hr"].positions, 1.0)

        # Periodic minimal-image distance
        d = lr_pos - hr_pos
        d = d - jnp.round(d)  # [-0.5, 0.5)
        pos_mse = jnp.mean(jnp.sum(d * d, axis=-1))
        val_pos_loss.append(pos_mse)

        lr_vel = val_batch["lr"].velocities
        hr_vel = val_batch["hr"].velocities
        dv = lr_vel - hr_vel
        vel_mse = jnp.mean(jnp.sum(dv * dv, axis=-1))
        val_vel_loss.append(vel_mse)

        pot_diff = val_batch["lr"].potential - val_batch["hr"].potential
        val_pot_loss.append(jnp.mean(pot_diff * pot_diff))

    logger.info(f"Positions MSE  = {float(sum(val_pos_loss) / len(val_pos_loss)):.5f}")
    logger.info(f"Velocities MSE = {float(sum(val_vel_loss) / len(val_vel_loss)):.5f}")
    logger.info(f"Potential MSE  = {float(sum(val_pot_loss) / len(val_pot_loss)):.5f}")


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
    Plots:
      1) 2D slab projection of delta (or log10(1+delta))
      2) Power spectrum ratio P(k)/P_HR(k)

    Robustness fixes:
      - Always deposit on LR mesh (mesh_plot = val_data["lr"].mesh)
      - Converts inputs to mesh units if they are in box units
      - Wrap positions to [0, mesh_plot) before CIC deposit
      - Power spectrum: forces numpy arrays, filters invalid bins (pk_hr<=0, NaNs/Infs)
    """
    max_idx = max_idx if max_idx is not None else -1
    mesh_plot = val_data["lr"].mesh  # always plot on LR mesh

    def to_mesh(pos, mesh_plot):
        """
        Convert positions to mesh units [0, mesh_plot).
        - If already mesh units (max ~ mesh_plot), do nothing.
        - If box units [0,1), scale by mesh_plot.
        """
        pos = np.asarray(pos)
        mx = np.max(pos)
        if mx <= 2.0:
            return pos * mesh_plot
        return pos

    # --- Convert to mesh units for deposit ---
    pos_pm = to_mesh(val_pos_pm[max_idx], mesh_plot)  # typically mesh units
    pos_hr = to_mesh(val_data["hr"].positions[max_idx], mesh_plot)  # may be box or mesh
    pos_lr = to_mesh(val_data["lr"].positions[max_idx], mesh_plot)

    # --- Wrap to [0, mesh_plot) to avoid out-of-range deposit artifacts ---
    pos_pm = np.mod(pos_pm, mesh_plot)
    pos_hr = np.mod(pos_hr, mesh_plot)
    pos_lr = np.mod(pos_lr, mesh_plot)

    # --- Deltas (density contrast) ---
    delta_pm = get_delta(pos_pm, (mesh_plot, mesh_plot, mesh_plot))
    delta_hr = get_delta(pos_hr, (mesh_plot, mesh_plot, mesh_plot))
    delta_lr = get_delta(pos_lr, (mesh_plot, mesh_plot, mesh_plot))

    # --- Slab projection around center (avoid full-box cancellation for delta) ---
    z0 = mesh_plot // 2
    h = max(1, slab_thickness // 2)
    sl = slice(max(z0 - h, 0), min(z0 + h, mesh_plot))

    proj_lr = delta_lr[:, :, sl].sum(axis=-1)
    proj_pm = delta_pm[:, :, sl].sum(axis=-1)
    proj_hr = delta_hr[:, :, sl].sum(axis=-1)

    # --- Visualization transform ---
    if plot_log:
        eps = 1e-4
        proj_lr = np.log10(np.clip(1.0 + proj_lr, eps, None))
        proj_pm = np.log10(np.clip(1.0 + proj_pm, eps, None))
        proj_hr = np.log10(np.clip(1.0 + proj_hr, eps, None))
        title_suffix = " (Log)"
    else:
        title_suffix = ""

    fig_delta, ax_delta = plt.subplots(ncols=3, figsize=(12, 5))
    cmap = "cividis"

    ax_delta[0].imshow(proj_lr, cmap=cmap)
    ax_delta[0].set_title(f"LR{title_suffix}", fontsize=20)

    ax_delta[1].imshow(proj_pm, cmap=cmap)
    ax_delta[1].set_title(f"LR + Nbodyify{title_suffix}", fontsize=20)

    ax_delta[2].imshow(proj_hr, cmap=cmap)
    ax_delta[2].set_title(f"HR{title_suffix}", fontsize=20)

    for ax in ax_delta:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()

    if use_wandb:
        import wandb

        wandb.log({f"{fig_label}_delta": wandb.Image(fig_delta)})
        plt.close(fig_delta)
    else:
        plt.show()

    # --- Power Spectrum (robust) ---
    def get_pk(delta):
        # Ensure CPU numpy arrays for power_spectrum / compensate_cic
        delta_np = np.asarray(delta)
        delta_np = compensate_cic(delta_np)
        k, pk = power_spectrum(
            delta_np,
            boxsize=np.array([box_size] * 3),
            kmin=np.pi / box_size,
            dk=2 * np.pi / box_size,
        )
        return np.asarray(k), np.asarray(pk)

    k, pk_hr = get_pk(delta_hr)
    _, pk_lr = get_pk(delta_lr)
    _, pk_pm = get_pk(delta_pm)

    # Filter invalid bins to avoid blank/NaN plots
    valid = (
        np.isfinite(k)
        & np.isfinite(pk_hr)
        & (pk_hr > 0)
        & np.isfinite(pk_lr)
        & np.isfinite(pk_pm)
    )
    k = k[valid]
    pk_hr = pk_hr[valid]
    pk_lr = pk_lr[valid]
    pk_pm = pk_pm[valid]

    fig_pk, ax_pk = plt.subplots(figsize=(8, 6))
    ax_pk.axhline(y=1, linestyle="dashed", color="black")  # ratio=1 reference

    if k.size > 0:
        ax_pk.semilogx(k, pk_lr / pk_hr, label="LR")
        ax_pk.semilogx(k, pk_pm / pk_hr, label="Nbodyify")
    else:
        ax_pk.text(
            0.5,
            0.5,
            "No valid P(k) bins (pk_hr<=0 or NaNs).",
            ha="center",
            va="center",
            transform=ax_pk.transAxes,
        )

    ax_pk.legend()
    ax_pk.set_xlabel(r"$k$ [$h \ \mathrm{Mpc}^{-1}$]")
    ax_pk.set_ylabel(r"$P(k)/P_\mathrm{HR}(k)$")
    ax_pk.set_title("Power spectrum ratio")

    if use_wandb:
        import wandb

        wandb.log({f"{fig_label}_pk": wandb.Image(fig_pk)})
        plt.close(fig_pk)
    else:
        plt.show()


# ==========================================
# Main Training Loop
# ==========================================


def train(config=None, data_dir=DEFAULT_DATA_DIR, output_dir=DEFAULT_MODEL_DIR):
    neural_net = build_network(config.correction_model)

    # Only potential-based losses need LR grid channels
    need_grid = config.training.loss in ("mse_potential", "mse_frozen_potential")

    cosmology, scale_factors, train_data, val_data, test_data = build_dataloader(
        config.data, data_dir=data_dir, need_grid=need_grid
    )

    print(
        f"Dataset summary: Train ({len(train_data)}), Val ({len(val_data)}), Test ({len(test_data)})"
    )

    params = initialize_network(
        train_data[0], neural_net=neural_net, model_type=config.correction_model.type
    )

    run = wandb.init(
        project=config.wandb.project, config=config.to_dict(), dir=output_dir
    )
    print(f"Run name: {run.name}")

    run_dir = output_dir / f"{run.name}"
    run_dir.mkdir(exist_ok=True, parents=True)

    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(config.to_dict(), f)

    # IMPORTANT: build_loss_fn must accept grid=None:
    # loss_fn(params, dataset, scale_factors, grid=None)
    loss_fn = build_loss_fn(
        config.training,
        neural_net,
        cosmology,
        correction_type=config.correction_model.type,
        mesh_lr=train_data[0]["lr"].mesh,
    )

    schedule = build_schedule(config.training.schedule)
    optimizer, opt_state = build_optimizer(
        config.training, params=params, schedule=schedule
    )

    print_initial_lr_loss(val_data)

    early_stop = EarlyStopping(patience=config.training.patience)
    best_params = params
    best_loss = float("inf")

    logger.info(f"need_grid = {need_grid}")

    # -----------------------
    # Loss wrapper (always receives grid; can be None)
    # -----------------------
    def train_loss_fn(p, batch, sf, grid):
        out = loss_fn(params=p, dataset=batch, scale_factors=sf, grid=grid)
        return out[0] if isinstance(out, tuple) else out

    @jax.jit
    def update_step(params, opt_state, batch, scale_factors, grid):
        train_loss, grads = jax.value_and_grad(train_loss_fn)(
            params, batch, scale_factors, grid
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return train_loss, params, opt_state

    pbar = tqdm(range(config.training.n_steps), desc="Training")
    eval_freq = 10 * config.training.batch_size

    for step in pbar:
        # --------- Train batch ---------
        batch = next(train_data.iterator)
        batch = train_data.move_to_device(
            batch, device=jax.devices()[0], build_grid=need_grid
        )

        # grid passed explicitly to JIT (or None)
        grid = batch["lr"].grid if need_grid else None
        if need_grid and grid is None:
            raise ValueError(
                "need_grid=True but batch['lr'].grid is None. Check dataloader and move_to_device(build_grid=True)."
            )

        train_loss, params, opt_state = update_step(
            params, opt_state, batch, scale_factors, grid
        )
        pbar.set_postfix({"Loss": float(jax.device_get(train_loss))})

        # --------- Validation ---------
        if step > 0 and step % eval_freq == 0:
            val_loss_device = jnp.zeros(())
            aux_first_batch = None
            val_batch_for_plot = None
            count = 0

            for i, val_batch in enumerate(val_data):
                val_batch = val_data.move_to_device(
                    val_batch, device=jax.devices()[0], build_grid=need_grid
                )
                grid_val = val_batch["lr"].grid if need_grid else None

                out = loss_fn(params, val_batch, scale_factors, grid=grid_val)

                if isinstance(out, tuple):
                    vl, aux = out
                else:
                    vl, aux = out, None

                val_loss_device += vl
                count += 1

                if i == 0:
                    val_batch_for_plot = val_batch
                    aux_first_batch = aux

            val_loss = float(jax.device_get(val_loss_device)) / max(count, 1)

            # Plot only if aux exists (positions loss returns aux; potential loss may not)
            if aux_first_batch is not None:
                aux_cpu = jax.device_get(aux_first_batch)
                mesh_plot = int(val_batch_for_plot["lr"].mesh)
                T = aux_cpu.shape[0]
                plot_indices = [0, T // 2, T - 1]

                for idx in plot_indices:
                    plot_eval(
                        aux_cpu,
                        val_batch_for_plot,
                        max_idx=idx,
                        fig_label=f"val_t{idx:02d}",
                        slab_thickness=64,
                        use_wandb=True,
                    )
            else:
                logger.info(
                    "Validation: loss_fn did not return aux; skipping plot_eval."
                )

            early_stop = early_stop.update(val_loss)
            if early_stop.has_improved:
                best_params = params
                best_loss = val_loss

            if hasattr(schedule, "step"):
                schedule.step(val_loss)

            learning_rate = opt_state.inner_opt_state[1].hyperparams["learning_rate"]
            train_loss_val = float(jax.device_get(train_loss))

            wandb.log(
                {
                    "train_loss": train_loss_val,
                    "val_loss": val_loss,
                    "learning_rate": float(learning_rate),
                },
                step=step,
            )

            pbar.set_postfix({"train_loss": train_loss_val, "val_loss": val_loss})

            if early_stop.should_stop:
                print(f"Early stopping triggered at step {step}.")
                break

        # --------- Checkpoints ---------
        if step > 0 and step % config.training.checkpoint_every == 0:
            checkpoint(
                run_dir=run_dir,
                loss=train_loss,
                params=params,
                prefix="train",
                step=step,
            )

    # -----------------------
    # Final evaluation
    # -----------------------
    checkpoint(run_dir=run_dir, params=best_params, loss=best_loss, prefix="best")

    test_batch = val_data.move_to_device(
        test_data[0], device=jax.devices()[0], build_grid=need_grid
    )
    grid_test = test_batch["lr"].grid if need_grid else None

    out = loss_fn(best_params, test_batch, scale_factors, grid=grid_test)
    if isinstance(out, tuple):
        test_loss, test_aux = out
    else:
        test_loss, test_aux = out, None

    print(f"Test loss = {float(jax.device_get(test_loss)):.5f}")

    if test_aux is not None:
        plot_eval(
            jax.device_get(test_aux),
            test_batch,
            max_idx=None,
            fig_label="test",
            use_wandb=True,
        )
    else:
        logger.info("Test: loss_fn did not return aux; skipping plot_eval.")

    wandb.finish()
    return best_loss


if __name__ == "__main__":
    FLAGS = flags.FLAGS
    FLAGS(sys.argv)

    cfg = load_config_yaml(FLAGS.config) if FLAGS.config else default_config()

    print("Running configuration:")
    import pprint

    pprint.pprint(cfg.to_dict())

    best_loss = train(cfg)
