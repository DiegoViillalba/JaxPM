
import jax
import jax.numpy as jnp
import numpy as np
from jaxpm.painting import cic_paint, cic_read, compensate_cic
from jaxpm.kernels import (
    fftk,
    laplace_kernel,
    longrange_kernel,
)
from jaxpm.utils import power_spectrum, cross_correlation_coefficients
from jax.experimental.ode import odeint
from jaxpm.pm import make_ode_fn, get_delta
import logging
logger = logging.getLogger(__name__)


def get_gravitational_potential(
    pos,
    n_mesh,
):
    mesh_shape = (n_mesh, n_mesh, n_mesh)
    kvec = fftk(mesh_shape)
    delta_k = jnp.fft.rfftn(cic_paint(jnp.zeros(mesh_shape), pos))
    pot_k = -delta_k * laplace_kernel(kvec) * longrange_kernel(kvec, r_split=0)
    pot_grid = 0.5 * jnp.fft.irfftn(pot_k)
    return pot_grid, cic_read(pot_grid, pos)


def get_frozen_potential_loss(
    neural_net,
):
    logger.info("JIT compiling frozen potential loss function.")
    @jax.jit
    def loss_fn(
        params,
        grid_data,
        pos_lr,
        potential_lr,
        potential_hr,
        scale_factors,
    ):
        if config.correction_model.type == "kcorr":
            kvec = fftk((grid_data.shape[:-1]))
            pot = grid_data[..., 0]
            pot_k = jnp.fft.rfftn(pot)
            kk = jnp.sqrt(sum((ki / jnp.pi) ** 2 for ki in kvec))
            net_output = neural_net.apply(
                params, kk, None, jnp.atleast_1d(scale_factors)
            )
            pot_k = pot_k * (1.0 + net_output)
            predicted_potential_grid = jnp.fft.irfftn(pot_k)
            predicted_potential = cic_read(predicted_potential_grid, pos_lr)
        elif config.correction_model.type == "cnn":
            corrected_potential = neural_net.apply(
                params,
                grid_data,
                pos_lr,
                scale_factors,
            )
            predicted_potential = corrected_potential.squeeze() + potential_lr
        return jnp.mean((predicted_potential - potential_hr) ** 2)

    return loss_fn


# def get_potential_loss(
#     neural_net,
#     cosmology,
#     correction_type,
# ):
#     logger.info("JIT compiling potential loss function.")
#     logger.info(f"Correction model type: {correction_type}")
#     @jax.jit
#     def loss_fn(
#         params,
#         grid_data,
#         pos_lr,
#         vel_lr,
#         potential_hr,
#         scale_factors,
#         correction_type=correction_type,
#     ):
#         n_mesh = grid_data.shape[1]

#         pos_pm, vel_pm = odeint(
#             make_ode_fn(
#                 mesh_shape=(n_mesh, n_mesh, n_mesh),
#                 add_correction=correction_type,
#                 model=neural_net,
#             ),
#             [pos_lr[0], vel_lr[0]],
#             scale_factors,
#             cosmology,
#             params,
#             rtol=1e-5,
#             atol=1e-5,
#         )
#         predicted_potential = jnp.stack(
#             [
#                 get_gravitational_potential(pos_pm[i], n_mesh)[1]
#                 for i in range(len(pos_pm))
#             ]
#         )
#         return jnp.mean((predicted_potential.squeeze() - potential_hr) ** 2)

#     return loss_fn

def get_potential_loss(neural_net, cosmology, correction_type):
    correction_type = str(correction_type)
    logger.info("JIT compiling potential loss function.")
    logger.info(f"Correction model type: {correction_type}")

    @jax.jit
    def loss_fn(params, grid_data, pos_lr, vel_lr, potential_hr, scale_factors):
        n_mesh = grid_data.shape[1]

        pos_pm, vel_pm = odeint(
            make_ode_fn(
                mesh_shape=(n_mesh, n_mesh, n_mesh),
                add_correction=correction_type,  
                model=neural_net,
            ),
            [pos_lr[0], vel_lr[0]],
            scale_factors,
            cosmology,
            params,
            rtol=1e-5,
            atol=1e-5,
        )

        predicted_potential = jnp.stack(
            [get_gravitational_potential(pos_pm[i], n_mesh)[1]
             for i in range(pos_pm.shape[0])]
        )

        loss = jnp.mean((predicted_potential.squeeze() - potential_hr) ** 2)

        return loss

    return loss_fn

def get_position_loss(
    neural_net,
    cosmology,
    n_mesh: int,
    lambda_pos=1.0,
    lambda_velocity=None,
    lambda_density=None,
    lambda_cross_corr=None,
    lambda_pk=None,
    correction_type=None,
    weight_snapshots=False,
    log_pos=False,
    fractional_mse=False,
):
    logger.info("JIT compiling position loss function.")
    @jax.jit
    def loss_fn(
        params,
        pos_lr,
        vel_lr,
        pos_hr,
        vel_hr,
        scale_factors,
    ):
        pos_pm, vel_pm = odeint(
            make_ode_fn(
                mesh_shape=(n_mesh, n_mesh, n_mesh),
                add_correction=correction_type,
                model=neural_net,
            ),
            [pos_lr[0], vel_lr[0]],
            scale_factors,
            cosmology,
            params,
            rtol=1e-5,
            atol=1e-5,
        )
        pos_pm %= n_mesh
        pos_hr %= n_mesh
        if weight_snapshots:
            snapshot_weights = (1.0 / scale_factors**1.7)[:, None]
        else:
            snapshot_weights = None
        sim_mse = lambda_pos * get_mse_pos(
            x=pos_pm,
            y=pos_hr,
            x_lr=pos_lr,
            box_size=n_mesh,
            snapshot_weights=snapshot_weights,
            apply_log=log_pos,
            fractional=fractional_mse,
        )
        if lambda_velocity not in (None, 0.0):
            sim_mse += lambda_velocity * jnp.mean(
                jnp.sum((vel_pm - vel_hr) ** 2, axis=-1)
            )
        if lambda_density not in (None, 0.0):
            sim_mse += lambda_density * get_density_loss(
                pos_pm, pos_hr, n_mesh_lr=n_mesh, n_mesh_hr=2 * n_mesh
            )
        if lambda_cross_corr not in (None, 0.0):
            sim_mse += lambda_cross_corr * get_cross_corr_loss(
                pos_pm, pos_hr, n_mesh_lr=n_mesh, n_mesh_hr=2 * n_mesh
            )
        if lambda_pk not in (None, 0.0):
            sim_mse += lambda_pk * get_pk_loss(
                pos_pm, pos_hr, n_mesh_lr=n_mesh, n_mesh_hr=2 * n_mesh
            )

        return sim_mse, pos_pm

    return loss_fn


def get_pk_loss(pos_pm, pos_hr, n_mesh_lr, n_mesh_hr, box_size=256.0):
    ratio = []
    for i in range(len(pos_pm)):
        delta_pm = get_delta(
            pos_pm[i] / n_mesh_lr * n_mesh_hr,
            mesh_shape=(n_mesh_hr, n_mesh_hr, n_mesh_hr),
        )
        delta_hr = get_delta(
            pos_hr[i] / n_mesh_lr * n_mesh_hr,
            mesh_shape=(n_mesh_hr, n_mesh_hr, n_mesh_hr),
        )
        k, pk_hr = power_spectrum(
            compensate_cic(delta_hr),
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
        ratio.append(pk_pm / pk_hr)

    return jnp.mean(jnp.sum((jnp.stack(ratio) - 1) ** 2, axis=-1))


def get_cross_corr_loss(pos_pm, pos_hr, n_mesh_lr, n_mesh_hr, box_size=256.0):
    cross_corrs = []
    for i in range(len(pos_pm)):
        delta_pm = get_delta(
            pos_pm[i] / n_mesh_lr * n_mesh_hr,
            mesh_shape=(n_mesh_hr, n_mesh_hr, n_mesh_hr),
        )
        delta_hr = get_delta(
            pos_hr[i] / n_mesh_lr * n_mesh_hr,
            mesh_shape=(n_mesh_hr, n_mesh_hr, n_mesh_hr),
        )
        k, pk_hr = power_spectrum(
            compensate_cic(delta_hr),
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
        cross_corrs.append(
            cross_correlation_coefficients(
                compensate_cic(delta_hr),
                compensate_cic(delta_pm),
                boxsize=np.array([box_size] * 3),
                kmin=np.pi / box_size,
                dk=2 * np.pi / box_size,
            )[1]
            / jnp.sqrt(pk_hr)
            / jnp.sqrt(pk_pm)
        )
    return -jnp.mean(jnp.stack(cross_corrs))


def get_density_loss(
    pos_pm,
    pos_hr,
    n_mesh_lr,
    n_mesh_hr,
):
    delta_pms, delta_hrs = [], []
    for i in range(len(pos_pm)):
        delta_pms.append(
            get_delta(
                pos_pm[i] / n_mesh_lr * n_mesh_hr,
                mesh_shape=(n_mesh_hr, n_mesh_hr, n_mesh_hr),
            )
        )
        delta_hrs.append(
            get_delta(
                pos_hr[i] / n_mesh_lr * n_mesh_hr,
                mesh_shape=(n_mesh_hr, n_mesh_hr, n_mesh_hr),
            )
        )
    delta_pms = jnp.stack(delta_pms)
    delta_hrs = jnp.stack(delta_hrs)
    return jnp.mean((delta_pms - delta_hrs) ** 2)


def get_mse_pos(
    x,
    y,
    x_lr,
    box_size=1.0,
    apply_log=False,
    snapshot_weights=None,
    fractional=False,
):
    dx = x - y
    if box_size is not None:
        dx = dx - box_size * jnp.round(dx / box_size)
    if apply_log:
        dx = jnp.log(jnp.abs(dx))
    if snapshot_weights is not None:
        return jnp.mean(snapshot_weights * jnp.sum(dx**2, axis=-1))
    elif fractional:
        dx_lr = x_lr - y
        if box_size is not None:
            dx_lr = dx_lr - box_size * jnp.round(dx_lr / box_size)
        return jnp.mean(jnp.sum(dx**2, axis=-1) / jnp.sum(dx_lr**2, axis=-1))
    return jnp.mean(jnp.sum(dx**2, axis=-1))
