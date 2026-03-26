from pathlib import Path
import os
import scienceplots
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import yaml
import jax.numpy as jnp
from jaxpm.nn import CNN, NeuralSplineFourierFilter

# from jaxpm.nn_utils import ReduceLROnPlateau
import sys
from absl import flags

##### deleted confg dependance

import jax
import optax
import haiku as hk
from tqdm import tqdm
from jax import config
import jax_cosmo as jc
from read_data import load_datasets
from flax.training.early_stopping import EarlyStopping
import wandb
import pickle

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "text.usetex": False,   # <- clave
    "font.family": "serif",
})

plt.style.use("default")
from jaxpm.painting import compensate_cic
from jaxpm.utils import power_spectrum
from jaxpm.pm import get_delta
from loss import (
    get_frozen_potential_loss,
    get_potential_loss,
    get_position_loss,
    get_mse_pos,
)

################Patch to not use config file
from typing import Any, Dict


class AttrDict(dict):
    """dict con acceso por atributos + to_dict() recursivo."""

    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return v

    def __setattr__(self, k, v):
        self[k] = v

    def to_dict(self):
        def conv(x):
            if isinstance(x, AttrDict):
                return {k: conv(v) for k, v in x.items()}
            if isinstance(x, dict):
                return {k: conv(v) for k, v in x.items()}
            return x

        return conv(self)


def to_attrdict(d: Dict[str, Any]) -> AttrDict:
    out = AttrDict()
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = to_attrdict(v)
        else:
            out[k] = v
    return out


def deep_update(base: dict, upd: dict) -> dict:
    """Merge recursivo: upd sobreescribe base."""
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def default_config() -> AttrDict:
    # Defaults razonables (ajusta a tu caso)
    cfg = {
  "data": {
    "mesh_lr": 128,
    "mesh_hr": 256,
    "n_train_sims": 80,
    "n_val_sims": 10,
    "snapshots": [10, 20, 30, 40, 49], 
    "box_size": 256.0,
    "n_snapshots": 50,
    "n_particles": 128,
  },
  "correction_model": {
    "type": "cnn",
    "channels_hidden_dim": 32,
    "n_convolutions": 4,
    "n_fully_connected": 2,
    "input_dim": 2,              # recomendado: [delta, phi] si tu pipeline lo soporta
    "kernel_size": 3,
    "pad_periodic": True,
    "embed_globals": False,       # para condicionar por a (si lo tienes)
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
    "n_steps": 1_000,
    "batch_size": 1,
    "patience": 30,
    "checkpoint_every": 1000,
    "sample_snapshots": True,
    "loss": "mse_positions",
    "weight_snapshots": True,
    "lambda_pos": 1.0,
    "lambda_velocity": 0.5,      # baja un poco si domina el gradiente
    "lambda_density": 0.1,       # ayuda a no “romper” el campo
    "lambda_pk": 0.05,           # si lo tienes implementado estable
    "lambda_cross_corr": 0.0,
    "log_pos": False,
    "fractional_mse": False,
    "weight_decay": 1e-4,
    "schedule": {
      "type": "cosine",
      "initial_lr": 0.0,
      "peak_value": 2e-4,
      "warmup_steps": 2000,
      "n_steps": 50_000,
      "min_lr": 1e-6,
      "factor": 0.5,
      "patience": 5,
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


flags.DEFINE_string("config", "", "Path a un YAML de configuración (opcional).")
#########################

# try velocity loss with pbcs (check range first)
# velocity validation? check field level emulator too
# add skip connections?
# Regenerate data for next bullet point
# k values -> how does it compare to lpt emulators?
# 5) Retrain frozen potential loss and compare potential predictions
# Compare to kcorr too
# 5) Run hopt with wandb
# 6) Add velocity loss, degeneracy with periodic boundaries?
# 7) Add flag for time diffusion embedding

config.update("jax_enable_x64", False)


def build_loss_fn(
    training_config,
    neural_net,
    cosmology,
    correction_type,
    mesh_lr: int,
):
    if training_config.loss == "mse_frozen_potential":
        single_loss_fn = get_frozen_potential_loss(
            neural_net=neural_net,
        )
        vmap_loss = jax.vmap(single_loss_fn, in_axes=(None, 0, 0, 0, 0, 0))

        def loss_fn(
            params,
            dataset,
            scale_factors,
        ):
            loss_array = vmap_loss(
                params,
                dataset["lr"].grid,
                dataset["lr"].positions * dataset["lr"].mesh,
                dataset["lr"].potential,
                dataset["hr"].potential,
                scale_factors,
            )
            return jnp.mean(loss_array)

    elif training_config.loss == "mse_potential":
        single_loss_fn = get_potential_loss(
            neural_net=neural_net,
            cosmology=cosmology,
        )

        def loss_fn(
            params,
            dataset,
            scale_factors,
        ):
            loss_array = single_loss_fn(
                params,
                dataset["lr"].grid,
                dataset["lr"].positions * dataset["lr"].mesh,
                dataset["lr"].velocities * dataset["lr"].mesh,
                dataset["hr"].potential,
                scale_factors,
            )
            return jnp.mean(loss_array)

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

        def loss_fn(
            params,
            dataset,
            scale_factors,
            max_idx,
        ):
            return single_loss_fn(
                params,
                dataset["lr"].positions[:max_idx] * dataset["lr"].mesh,
                dataset["lr"].velocities[:max_idx] * dataset["lr"].mesh,
                dataset["hr"].positions[:max_idx] * dataset["lr"].mesh,
                dataset["hr"].velocities[:max_idx] * dataset["lr"].mesh,
                scale_factors[:max_idx],
            )

    return loss_fn


def build_network(
    config,
):
    def CNNCorr(
        x,
        positions,
        scale_factors,
        velocities,
    ):
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

    def KCorr(
        x,
        scale_factors,
    ):
        return NeuralSplineFourierFilter(
            n_knots=config.n_knots, latent_size=config.latent_size
        )(x, scale_factors)

    if config.type == "cnn" or config.type == "cnn_force":
        return hk.without_apply_rng(hk.transform(CNNCorr))
    elif config.type == "kcorr":
        return hk.without_apply_rng(hk.transform(KCorr))
    elif config.type == "cnn+kcorr":
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
    rng = jax.random.PRNGKey(seed)
    grid_input_init = data_sample["lr"].grid[0]
    pos_init = data_sample["lr"].positions[0]
    vel_init = data_sample["lr"].velocities[0]
    scale_init = jnp.array(1.0)
    if model_type == "kcorr":
        params = neural_net.init(
            rng,
            grid_input_init,
            scale_init,
        )
    elif model_type == "cnn" or model_type == "cnn_force":
        params = neural_net.init(
            rng,
            grid_input_init,
            pos_init,
            scale_init,
            vel_init,
        )
    elif model_type == "cnn+kcorr":
        params = {}
        params["kcorr"] = neural_net["kcorr"].init(
            rng,
            grid_input_init,
            scale_init,
        )
        params["cnn"] = neural_net["cnn"].init(
            rng, grid_input_init, pos_init, scale_init, None
        )
    return params


def build_dataloader(
    config,
    data_dir=Path("/cosmos_storage/home/diegovillalba/JaxPM/data/"),
):
    omega_c = 0.25
    sigma8 = 0.8
    cosmology = jc.Planck15(Omega_c=omega_c, sigma8=sigma8)
    mesh_lr = config.mesh_lr
    mesh_hr = config.mesh_hr
    n_train_sims = config.n_train_sims
    n_val_sims = config.n_val_sims
    snapshots = config.snapshots
    box_size = config.box_size
    n_snapshots = config.n_snapshots
    n_particles = config.n_particles
    data_dir /= (
        f"matched_{mesh_lr}_{mesh_hr}_L{box_size:.1f}_S{n_snapshots}_Np{n_particles}/"
    )
    scale_factors = jnp.load(data_dir / "scale_factors.npy")
    if snapshots is not None:
        snapshots = jnp.array(snapshots)
        scale_factors = scale_factors[snapshots]
    train_data, val_data, test_data = load_datasets(
        n_train_sims,
        n_val_sims,
        1,
        mesh_hr=mesh_hr,
        mesh_lr=mesh_lr,
        data_dir=data_dir,
        snapshots=snapshots,
        box_size=box_size,
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
    if config.type == "plateau":
        return ReduceLROnPlateau(
            initial_lr=config.initial_lr,
            factor=config.factor,
            patience=config.patience,
            min_lr=config.min_lr,
        )


def build_optimizer(
    config,
    params,
    schedule=None,
):
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


def print_initial_lr_loss(
    val_data,
):
    val_pos_loss, val_vel_loss, val_pot_loss = [], [], []
    for val_batch in val_data:
        val_pos_loss.append(
            get_mse_pos(
                val_batch["hr"].positions * val_batch["lr"].mesh,
                val_batch["lr"].positions * val_batch["lr"].mesh,
                x_lr=val_batch["lr"].positions * val_batch["lr"].mesh,
                box_size=val_batch["lr"].mesh,
            )
        )
        val_vel_loss.append(
            jnp.mean(
                (
                    val_batch["lr"].velocities * val_batch["lr"].mesh
                    - val_batch["hr"].velocities * val_batch["lr"].mesh
                )
                ** 2
            )
        )
        val_pot_loss.append(
            jnp.mean((val_batch["lr"].potential - val_batch["hr"].potential) ** 2)
        )
    print("Positions MSE = ", sum(val_pos_loss) / len(val_pos_loss))
    print("Velocities MSE = ", sum(val_vel_loss) / len(val_vel_loss))
    print("Potential MSE = ", sum(val_pot_loss) / len(val_pot_loss))


def checkpoint(run_dir, loss, params, prefix, step=None):
    if step is not None:
        filename = f"{prefix}_{loss:.3f}_weights_{step}.pkl"
    else:
        filename = f"{prefix}_{loss:.3f}_weights.pkl"
    with open(run_dir / filename, "wb") as f:
        state_dict = hk.data_structures.to_immutable_dict(params)
        pickle.dump(state_dict, f)


def plot_eval(
    val_pos_pm,
    val_data,
    max_idx=None,
    box_size=256.0,
    fig_label="val",
    step=None, 
    **kwargs
):
    if max_idx is None:
        max_idx = -1
    mesh_plot = val_data["hr"].mesh
    delta_pm = get_delta(
        val_pos_pm[max_idx] / val_data["lr"].mesh * mesh_plot,
        (mesh_plot, mesh_plot, mesh_plot),
    )
    delta_hr = get_delta(
        val_data["hr"].positions[max_idx] * mesh_plot, (mesh_plot, mesh_plot, mesh_plot)
    )
    delta_lr = get_delta(
        val_data["lr"].positions[max_idx] * mesh_plot, (mesh_plot, mesh_plot, mesh_plot)
    )
    fig, ax = plt.subplots(ncols=3, figsize=(12, 5))
    cmap = "cividis"
    ax[0].imshow(delta_lr[:, :, :5].sum(axis=-1), cmap=cmap)
    ax[1].imshow(delta_pm[:, :, :5].sum(axis=-1), cmap=cmap)
    ax[2].imshow(delta_hr[:, :, :5].sum(axis=-1), cmap=cmap)
    ax[0].set_title(
        "LR",
        fontsize=20,
    )
    ax[1].set_title(
        "LR + Nbodyify",
        fontsize=20,
    )
    ax[2].set_title(
        "HR",
        fontsize=20,
    )
    for a in ax:
        a.set_xticks([])  # Remove x-ticks and labels
        a.set_yticks([])
    plt.tight_layout()
    wandb.log({f"{fig_label}_delta": plt})
    plt.close()
    k, pk_hr = power_spectrum(
        compensate_cic(delta_hr),
        boxsize=np.array([box_size] * 3),
        kmin=np.pi / box_size,
        dk=2 * np.pi / box_size,
    )

    k, pk_lr = power_spectrum(
        compensate_cic(delta_lr),
        boxsize=np.array([box_size] * 3),
        kmin=np.pi / box_size,
        dk=2 * np.pi / box_size,
    )

    k, pk_pm = power_spectrum(
        compensate_cic(delta_pm),
        boxsize=np.array([box_size] * 3),
        kmin=np.pi / box_size,
        dk=2 * np.pi / box_size,
    )

    plt.axhline(y=0, linestyle="dashed", color="black")
    plt.semilogx(
        k,
        pk_lr / pk_hr,
        label="LR",
    )
    plt.semilogx(
        k,
        pk_pm / pk_hr,
        label="Nbodyify",
    )
    plt.legend()
    plt.xlabel(r"$k$ [$h \ \mathrm{Mpc}^{-1}$]")
    plt.ylabel(r"$P(k)/P_\mathrm{HR}(k)$")
    fig = plt.gcf()  # o el fig explícito
    # wandb.log({f"{fig_label}_pk": wandb.Image(fig)}, step=global_step)
    if step is None:
        wandb.log({f"{fig_label}_pk": wandb.Image(fig)})
    else:
        wandb.log({f"{fig_label}_pk": wandb.Image(fig)}, step=int(step))    
    plt.close(fig)


def train_step(
    *,
    train_data,
    val_data,
    scale_factors,
    loss_fn,
    optimizer,
    opt_state,
    params,
    best_params,
    early_stop,
    step,
    global_step,   # <- AÑADIR
    pbar,
    schedule,
    max_idx,
):
    # def train_loss_fn(
    #     params,
    # ):
    #     batch = next(train_data.iterator)
    #     batch = train_data.move_to_device(batch, device=jax.devices()[0])
    #     return loss_fn(
    #         params=params,
    #         dataset=batch,
    #         scale_factors=scale_factors,
    #         max_idx=max_idx,
    #     )[0]

    # train_loss, grads = jax.value_and_grad(
    #     train_loss_fn,
    # )(
    #     params,
    # )
    batch = next(train_data.iterator)
    batch = train_data.move_to_device(batch, device=jax.devices()[0])

    def train_loss_fn(params, batch, scale_factors, max_idx):
        return loss_fn(
            params=params, dataset=batch, scale_factors=scale_factors, max_idx=max_idx
        )[0]

    train_loss, grads = jax.value_and_grad(train_loss_fn)(
        params, batch, scale_factors, max_idx
    )
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    pbar.set_postfix({"Step": global_step, "Loss": float(train_loss)})

    # ---- frecuencia de validación/plots/logging (ARREGLADA) ----
    eval_every = 10 * config.training.batch_size  # o lo que tú quieras
    do_eval = (global_step % eval_every) == 0

    if do_eval:
        val_loss = 0.0
        last_val_batch = None
        val_pos_pm = None

        for val_batch in val_data:
            last_val_batch = val_data.move_to_device(val_batch, device=jax.devices()[0])
            vl, val_pos_pm = loss_fn(
                params,
                last_val_batch,
                scale_factors,
                max_idx=max_idx,
            )
            val_loss += vl

        val_loss /= len(val_data)

        # plot cada N evaluaciones (por ejemplo cada eval)
        # si quieres menos frecuente: plot_every = eval_every * 10, etc.
        plot_every = eval_every  # ajusta
        if (global_step % plot_every) == 0:
            plot_eval(val_pos_pm, last_val_batch, max_idx=max_idx, step=global_step)

        # EARLY STOP: solo una llamada por evaluación
        # asumo que update devuelve algo tipo (has_improved, should_stop)
        # Si tu clase no lo hace, ajusta a tu API.
        # has_improved, should_stop = early_stop.update(val_loss)
        
        out = early_stop.update(val_loss)

        # Caso 1: update() devuelve bool (has_improved)
        if isinstance(out, bool):
            has_improved = out
            # early_stop sigue siendo el objeto original
            should_stop = getattr(early_stop, "should_stop", False)

        # Caso 2: update() devuelve el objeto (self) u otro objeto
        else:
            early_stop = out
            has_improved = getattr(early_stop, "has_improved", False)
            should_stop  = getattr(early_stop, "should_stop", False)

        # Normaliza por si vienen como numpy/jax scalars
        has_improved = bool(has_improved)
        should_stop  = bool(should_stop)

        if has_improved:
            best_params = params

        if hasattr(schedule, "step"):
            schedule.step(val_loss)

        learning_rate = opt_state.inner_opt_state[1].hyperparams["learning_rate"]

        wandb.log(
            {
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "learning_rate": float(learning_rate),
            },
            step=int(global_step),
        )

        pbar.set_postfix(val_loss=float(val_loss))

    else:
        should_stop = False

    return train_loss, params, best_params, opt_state, early_stop, should_stop

# def train(
#     config=None,
#     data_dir=Path("/cosmos_storage/home/diegovillalba/JaxPM/data/"),
#     output_dir=Path("/cosmos_storage/home/diegovillalba/JaxPM/models/"),
# ):
#     neural_net = build_network(
#         config.correction_model,
#     )
#     cosmology, scale_factors, train_data, val_data, test_data = build_dataloader(
#         config.data,
#         data_dir=data_dir,
#     )
#     print(f"Using {len(train_data)} sims for training")
#     print(f"Using {len(val_data)} sims for val")
#     print(f"Using {len(test_data)} sims for test")
#     params = initialize_network(
#         train_data[0], neural_net=neural_net, model_type=config.correction_model.type
#     )

#     run = wandb.init(
#         project=config.wandb.project,
#         config=config.to_dict(),
#         dir=output_dir,
#     )
#     wandb.config = config
#     print(f"Run name: {run.name}")
#     run_dir = output_dir / f"{run.name}"
#     run_dir.mkdir(exist_ok=True, parents=True)
#     with open(run_dir / "config.yaml", "w") as f:
#         yaml.dump(config.to_dict(), f)

#     loss_fn = build_loss_fn(
#         config.training,
#         neural_net,
#         cosmology,
#         correction_type=config.correction_model.type,
#         mesh_lr=train_data[0]["lr"].mesh,
#     )
#     schedule = build_schedule(config.training.schedule)
#     optimizer, opt_state = build_optimizer(
#         config.training,
#         params=params,
#         schedule=schedule,
#     )

#     print_initial_lr_loss(
#         val_data,
#     )
#     early_stop = EarlyStopping(patience=config.training.patience)
#     best_params = None
#     pbar = tqdm(range(config.training.n_steps))
#     rng = jax.random.PRNGKey(0)
#     for step in pbar:
#         if config.training.sample_snapshots:
#             rng, _ = jax.random.split(rng)
#             max_idx = jax.random.randint(
#                 rng, minval=10, maxval=len(scale_factors), shape=(1,)
#             )[0]
#         else:
#             max_idx = None
#         (
#             train_loss,
#             params,
#             best_params,
#             opt_state,
#             early_stop,
#             should_stop,
#         ) = train_step(
#             train_data=train_data,
#             val_data=val_data,
#             scale_factors=scale_factors,
#             loss_fn=loss_fn,
#             optimizer=optimizer,
#             opt_state=opt_state,
#             params=params,
#             best_params=best_params,
#             early_stop=early_stop,
#             step=step,
#             pbar=pbar,
#             schedule=schedule,
#             max_idx=max_idx,
#         )
#         if should_stop:
#             break
#         if step % config.training.checkpoint_every == 0:
#             checkpoint(
#                 run_dir=run_dir,
#                 loss=train_loss,
#                 params=params,
#                 prefix="train",
#                 step=step,
#             )
#     best_loss = early_stop.best_metric
#     checkpoint(run_dir=run_dir, params=best_params, loss=best_loss, prefix="best")
#     test_batch = val_data.move_to_device(test_data[0], device=jax.devices()[0])
#     test_loss, test_pos_pm = loss_fn(
#         best_params,
#         test_batch,
#         scale_factors,
#         max_idx=None,
#     )
#     print(f"Test loss = {test_loss:.5f}")
#     plot_eval(test_pos_pm, test_batch, max_idx=None, fig_label="test")
#     return best_loss

def train(
    config=None,
    data_dir=Path("/cosmos_storage/home/diegovillalba/JaxPM/data/"),
    output_dir=Path("/cosmos_storage/home/diegovillalba/JaxPM/models/"),
):
    # ----------------------------
    # Build model + data
    # ----------------------------
    neural_net = build_network(config.correction_model)

    cosmology, scale_factors, train_data, val_data, test_data = build_dataloader(
        config.data,
        data_dir=data_dir,
    )

    print(f"Using {len(train_data)} sims for training")
    print(f"Using {len(val_data)} sims for val")
    print(f"Using {len(test_data)} sims for test")

    params = initialize_network(
        train_data[0],
        neural_net=neural_net,
        model_type=config.correction_model.type,
    )

    # ----------------------------
    # W&B init
    # ----------------------------
    run = wandb.init(
        project=config.wandb.project,
        config=config.to_dict(),   # aquí va el dict
        dir=str(output_dir),
    )

    print(f"Run name: {run.name}")
    run_dir = output_dir / f"{run.name}"
    run_dir.mkdir(exist_ok=True, parents=True)

    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config.to_dict(), f, sort_keys=False)

    # ----------------------------
    # Loss + optimizer
    # ----------------------------
    loss_fn = build_loss_fn(
        config.training,
        neural_net,
        cosmology,
        correction_type=config.correction_model.type,
        mesh_lr=train_data[0]["lr"].mesh,
    )

    schedule = build_schedule(config.training.schedule)
    optimizer, opt_state = build_optimizer(
        config.training,
        params=params,
        schedule=schedule,
    )

    print_initial_lr_loss(val_data)

    early_stop = EarlyStopping(patience=config.training.patience)

    # Si nunca "mejora", al menos tenemos algo válido
    best_params = params

    # ----------------------------
    # Train loop
    # ----------------------------
    pbar = tqdm(range(config.training.n_steps))
    rng = jax.random.PRNGKey(0)

    global_step = 0  # <- clave para W&B monotónico

    for step in pbar:
        if config.training.sample_snapshots:
            rng, _ = jax.random.split(rng)
            max_idx = jax.random.randint(
                rng, minval=10, maxval=len(scale_factors), shape=(1,)
            )[0]
        else:
            max_idx = None

        (
            train_loss,
            params,
            best_params,
            opt_state,
            early_stop,
            should_stop,
        ) = train_step(
            train_data=train_data,
            val_data=val_data,
            scale_factors=scale_factors,
            loss_fn=loss_fn,
            optimizer=optimizer,
            opt_state=opt_state,
            params=params,
            best_params=best_params,
            early_stop=early_stop,
            step=step,
            global_step=global_step,   # <- NUEVO
            pbar=pbar,
            schedule=schedule,
            max_idx=max_idx,
        )

        global_step += 1

        if should_stop:
            break

        if (step % config.training.checkpoint_every) == 0:
            checkpoint(
                run_dir=run_dir,
                loss=float(train_loss),
                params=params,
                prefix="train",
                step=step,
            )

    # ----------------------------
    # Save best + test
    # ----------------------------
    best_loss = early_stop.best_metric
    checkpoint(
        run_dir=run_dir,
        params=best_params,
        loss=float(best_loss) if best_loss is not None else None,
        prefix="best",
    )

    test_batch = test_data.move_to_device(test_data[0], device=jax.devices()[0])  # <- FIX
    test_loss, test_pos_pm = loss_fn(
        best_params,
        test_batch,
        scale_factors,
        max_idx=None,
    )

    print(f"Test loss = {float(test_loss):.5f}")

    # Si plot_eval loggea a wandb, pásale step global o uno fijo:
    # plot_eval(test_pos_pm, test_batch, max_idx=None, fig_label="test", step=global_step)
    plot_eval(test_pos_pm, test_batch, max_idx=None, fig_label="test")

    run.finish()
    return best_loss

if __name__ == "__main__":
    FLAGS = flags.FLAGS
    FLAGS(sys.argv)

    if FLAGS.config:
        config = load_config_yaml(FLAGS.config)
    else:
        config = default_config()

    print("Running configuration")
    print(config.to_dict())

    best_loss = train(config)

# # Copiar config original
# cp /cosmos_storage/home/diegovillalba/JaxPM/models/electric-dream-78/config.yaml \
#    /cosmos_storage/home/diegovillalba/JaxPM/models/electric_dream_bugfix.yaml

# # Lanzar
# python train_refactored.py --config /cosmos_storage/home/diegovillalba/JaxPM/models/electric_dream_bugfix.yaml


# nohup python train_refactored.py \
#   --config /cosmos_storage/home/diegovillalba/JaxPM/models/electric_dream_bugfix.yaml\
#   > /cosmos_storage/home/diegovillalba/JaxPM/logs/electric_dream_bugfix.log 2>&1 &

# echo "PID: $!"