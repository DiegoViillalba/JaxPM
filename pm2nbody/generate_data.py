from pathlib import Path
import scipy
import jax
import jax.numpy as jnp
import jax_cosmo as jc
from jax.experimental.ode import odeint
from jaxpm.pm import linear_field, lpt, make_ode_fn, get_delta
from jaxpm.kernels import fftk, longrange_kernel, laplace_kernel
from jaxpm.painting import cic_paint, cic_read
from functools import partial
import numpy as np
from jax import config

config.update("jax_enable_x64", True)


# def get_linear_field(mesh_shape, box_size, omega_c, sigma8, seed=0):
#     k = jnp.logspace(-4, 1, 128)
#     pk = jc.power.linear_matter_power(jc.Planck15(Omega_c=omega_c, sigma8=sigma8), k)
#     pk_fn = lambda x: jc.scipy.interpolate.interp(x.reshape([-1]), k, pk).reshape(
#         x.shape
#     )
#     return linear_field(mesh_shape, box_size, pk_fn, seed=jax.random.PRNGKey(seed))


def downsample_field(
    field: jnp.array,
    downsampling_factor: int = 2,
):
    filter_size = (downsampling_factor, downsampling_factor, downsampling_factor)
    filter_weights = np.ones(filter_size) / np.prod(filter_size)
    result = scipy.ndimage.convolve(field, filter_weights, mode="mirror")
    result = result[::downsampling_factor, ::downsampling_factor, ::downsampling_factor]
    return result


def get_linear_field(mesh_shape, box_size, omega_c, sigma8, seed=0):
    # tabulación de P(k)
    k = jnp.logspace(-4, 1, 128)
    pk = jc.power.linear_matter_power(jc.Planck15(Omega_c=omega_c, sigma8=sigma8), k)

    def pk_fn(x):
        # x: array (kmesh) con shape 3D
        xflat = x.reshape(-1)

        # índices de interpolación lineal
        idx = jnp.searchsorted(k, xflat)
        idx = jnp.clip(idx, 1, k.size - 1)

        k0 = k[idx - 1]
        k1 = k[idx]
        p0 = pk[idx - 1]
        p1 = pk[idx]

        t = (xflat - k0) / (k1 - k0 + 1e-12)
        out = p0 + t * (p1 - p0)
        return out.reshape(x.shape)

    # Si sigues viendo OOM, fuerza esta línea a CPU:
    # with jax.default_device(jax.devices("cpu")[0]):
    return linear_field(mesh_shape, box_size, pk_fn, seed=jax.random.PRNGKey(seed))


def arange_particles_in_mesh(
    n_particles,
    mesh_shape,
):
    n_particles_per_side = jnp.ceil(n_particles ** (1 / 3))
    return (
        jnp.stack(
            jnp.meshgrid(
                *[jnp.arange(n_particles_per_side) * mesh_shape[s] for s in range(3)]
            ),
            axis=-1,
        ).reshape([-1, 3])
        / n_particles_per_side
    )


def get_ics(
    n_particles,
    mesh_shape,
    linear_field,
    snapshot,
    omega_c,
    sigma8,
):
    # Create particles
    particles = arange_particles_in_mesh(
        n_particles=n_particles,
        mesh_shape=mesh_shape,
    )
    cosmo = jc.Planck15(Omega_c=omega_c, sigma8=sigma8)
    # Initial displacement
    dx, p, f = lpt(cosmo, linear_field, particles, snapshot)
    return (particles + dx, p)


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


def get_density(
    pos,
    n_mesh,
):
    grid_dens = get_delta(pos, (n_mesh, n_mesh, n_mesh))
    dens = cic_read(grid_dens, pos)
    return grid_dens, dens


@partial(jax.jit, static_argnums=(0))
def run_simulation(
    n_mesh,
    omega_c,
    sigma8,
    initial_conditions,
):
    cosmo = jc.Planck15(Omega_c=omega_c, sigma8=sigma8)
    return odeint(
        make_ode_fn((n_mesh, n_mesh, n_mesh)),
        initial_conditions,
        snapshots,
        cosmo,
        rtol=1e-5,
        atol=1e-5,
    )


if __name__ == "__main__":
    out_dir = Path("/home/jvazquez/diego_villalba/florpi/JaxPM/data")
    mesh_lr = 64  # 128
    mesh_hr = 128  # 256 #128
    n_particles_sqrt_3 = mesh_lr
    n_particles = n_particles_sqrt_3**3  # mesh_hr**3
    n_snapshots = 10
    snapshots = jnp.linspace(0.1, 1.0, n_snapshots)
    L = 256.0
    out_dir /= (
        f"matched_{mesh_lr}_{mesh_hr}_L{L:.1f}_S{n_snapshots}_Np{n_particles_sqrt_3}"
    )
    out_dir.mkdir(exist_ok=True, parents=True)
    mesh_shape_hr = (mesh_hr, mesh_hr, mesh_hr)
    mesh_shape_lr = (mesh_lr, mesh_lr, mesh_lr)
    box_size = [L, L, L]
    omega_c = 0.25
    sigma8 = 0.8
    ics_seed = 0
    n_sims = 3
    for n in range(n_sims):
        # Generate density field ICs
        print("*" * 10)
        print(n)
        linear_hr = get_linear_field(
            mesh_shape=mesh_shape_hr,
            box_size=box_size,
            omega_c=omega_c,
            sigma8=sigma8,
            seed=n,
        )
        linear_lr = downsample_field(
            linear_hr,
            downsampling_factor=mesh_hr // mesh_lr,
        )
        ics_hr = get_ics(
            n_particles=n_particles,
            mesh_shape=mesh_shape_hr,
            linear_field=linear_hr,
            snapshot=snapshots[0],
            omega_c=omega_c,
            sigma8=sigma8,
        )
        ics_lr = get_ics(
            n_particles=n_particles,
            mesh_shape=mesh_shape_lr,
            linear_field=linear_lr,
            snapshot=snapshots[0],
            omega_c=omega_c,
            sigma8=sigma8,
        )

        # Run simulations
        pos_hr, vel_hr = run_simulation(
            mesh_hr,
            omega_c=omega_c,
            sigma8=sigma8,
            initial_conditions=ics_hr,
        )
        pos_lr, vel_lr = run_simulation(
            mesh_lr,
            omega_c=omega_c,
            sigma8=sigma8,
            initial_conditions=ics_lr,
        )

        pot_hr, pot_lr, pot_grid_lr = [], [], []
        for s in range(len(snapshots)):
            _, phr = get_gravitational_potential(pos_hr[s], mesh_hr)
            pot_hr.append(phr)
            pgridlr, plr = get_gravitational_potential(pos_lr[s], mesh_lr)
            pot_lr.append(plr)
            pot_grid_lr.append(pgridlr)
        pot_hr = jnp.stack(pot_hr)
        pot_lr = jnp.stack(pot_lr)
        pot_grid_lr = jnp.stack(pot_grid_lr)
        jnp.save(out_dir / f"pos_m{mesh_hr}_s{n}.npy", pos_hr / mesh_hr * L)
        jnp.save(out_dir / f"vel_m{mesh_hr}_s{n}.npy", vel_hr / mesh_hr * L)
        jnp.save(out_dir / f"pot_m{mesh_hr}_s{n}.npy", pot_hr)
        jnp.save(out_dir / f"pos_m{mesh_lr}_s{n}.npy", pos_lr / mesh_lr * L)
        jnp.save(out_dir / f"vel_m{mesh_lr}_s{n}.npy", vel_lr / mesh_lr * L)
        jnp.save(out_dir / f"pot_m{mesh_lr}_s{n}.npy", pot_lr)
        jnp.save(out_dir / f"pot_grid_m{mesh_lr}_s{n}.npy", pot_grid_lr)
        jnp.save(out_dir / f"scale_factors.npy", snapshots)
        del vel_hr, pot_hr, ics_hr
        dens_hr, dens_lr, dens_grid_lr = [], [], []
        for s in range(len(snapshots)):
            _, dhr = get_density(pos_hr[s], mesh_hr)
            dens_hr.append(dhr)
            dgridlr, dlr = get_density(pos_lr[s], mesh_lr)
            dens_lr.append(dlr)
            dens_grid_lr.append(dgridlr)
        dens_hr = jnp.stack(dens_hr)
        dens_lr = jnp.stack(dens_lr)
        dens_grid_lr = jnp.stack(dens_grid_lr)
        jnp.save(out_dir / f"dens_m{mesh_hr}_s{n}.npy", dens_hr)
        jnp.save(out_dir / f"dens_m{mesh_lr}_s{n}.npy", dens_lr)
        jnp.save(out_dir / f"dens_grid_m{mesh_lr}_s{n}.npy", dens_grid_lr)
        del pos_hr, dens_hr
