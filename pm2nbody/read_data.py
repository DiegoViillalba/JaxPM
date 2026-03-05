from pathlib import Path
import jax
from typing import Optional, List
from dataclasses import dataclass
import jax.numpy as jnp
import numpy as np
import logging
logger = logging.getLogger(__name__)

def downsample_to_mesh(
    array: jnp.array, n_mesh: int, downsampling_factor: int
) -> jnp.array:
    """Downsample an array to the number of particles in a lower resolution mesh, such
    that low and high resolution particles match

    """
    first_dim_array = len(array)
    if len(array.shape) == 3:
        last_dim_array = array.shape[-1]
        first_reshape_to = (first_dim_array, n_mesh, n_mesh, n_mesh, last_dim_array)
        last_reshape_to = (first_dim_array, -1, last_dim_array)
    elif len(array.shape) == 2:
        first_reshape_to = (
            first_dim_array,
            n_mesh,
            n_mesh,
            n_mesh,
        )
        last_reshape_to = (first_dim_array, -1)
    else:
        raise ValueError("Array must be 2 or 3 dimensional")
    return array.reshape(first_reshape_to)[
        :,
        ::downsampling_factor,
        ::downsampling_factor,
        ::downsampling_factor,
    ].reshape(last_reshape_to)



def get_data(
    data_dir: Path,
    n_mesh: int,
    downsampling_factor: Optional[int] = None,
    get_grids: Optional[bool] = False,
    snapshots: Optional[List[int]] = None,
    box_size: Optional[float] = 256.0,
    normalize_to_box: Optional[bool] = True,
    idx: Optional[int] = 0,
    move_to_cpu: bool = True,
):
    #Load on CPU with numpy
    pos = np.load(data_dir / f"pos_m{n_mesh}_s{idx}.npy")          # numpy array (CPU)

    if snapshots is None:
        snapshots = np.arange(len(pos))                           # numpy indices (CPU)
    else:
        snapshots = np.asarray(snapshots, dtype=np.int64)

    # Slice on CPU
    pos = pos[snapshots, :, :]

    vel = np.load(data_dir / f"vel_m{n_mesh}_s{idx}.npy")[snapshots]

    if normalize_to_box:
        pos = pos / box_size
        vel = vel / box_size

    gravitational_potential = np.load(data_dir / f"pot_m{n_mesh}_s{idx}.npy")[snapshots]

    if downsampling_factor is not None:
        # If downsample_to_mesh expects jnp arrays, convert locally (still CPU)
        pos = downsample_to_mesh(array=jnp.asarray(pos), n_mesh=n_mesh, downsampling_factor=downsampling_factor)
        vel = downsample_to_mesh(array=jnp.asarray(vel), n_mesh=n_mesh, downsampling_factor=downsampling_factor)
        gravitational_potential = downsample_to_mesh(array=jnp.asarray(gravitational_potential), n_mesh=n_mesh, downsampling_factor=downsampling_factor)
    else:
        # keep as numpy for now
        pass

    if not get_grids:
        # return as jnp on CPU if you want consistency downstream
        pos = jnp.asarray(pos)
        vel = jnp.asarray(vel)
        gravitational_potential = jnp.asarray(gravitational_potential, dtype=jnp.float32)

        if move_to_cpu:
            cpu = jax.devices("cpu")[0]
            pos = jax.device_put(pos, cpu)
            vel = jax.device_put(vel, cpu)
            gravitational_potential = jax.device_put(gravitational_potential, cpu)

        return pos, vel, gravitational_potential

    potential_grid = np.load(data_dir / f"pot_grid_m{n_mesh}_s{idx}.npy")[snapshots]
    density_grid   = np.load(data_dir / f"dens_grid_m{n_mesh}_s{idx}.npy")[snapshots]

    # Convert to jax arrays (still CPU unless you device_put elsewhere)
    pos = jnp.asarray(pos)
    vel = jnp.asarray(vel)
    gravitational_potential = jnp.asarray(gravitational_potential)
    potential_grid = jnp.asarray(potential_grid)
    density_grid = jnp.asarray(density_grid)

    if move_to_cpu:
        cpu = jax.devices("cpu")[0]
        pos = jax.device_put(pos, cpu)
        vel = jax.device_put(vel, cpu)
        gravitational_potential = jax.device_put(gravitational_potential, cpu)
        potential_grid = jax.device_put(potential_grid, cpu)
        density_grid = jax.device_put(density_grid, cpu)

    return pos, vel, gravitational_potential, potential_grid, density_grid


@jax.tree_util.register_pytree_node_class
@dataclass
class ResolutionData:
    mesh: int
    positions: jnp.ndarray
    velocities: jnp.ndarray
    potential: jnp.ndarray
    potential_grid: Optional[jnp.ndarray] = None
    density_grid: Optional[jnp.ndarray] = None

    # Derived (never traced / never stored)
    grid: Optional[jnp.ndarray] = None
    _scaled_to_mesh: bool = False  # metadata only

    def __post_init__(self):
        self.grid = None

    def get_grid(self):
        if self.potential_grid is None or self.density_grid is None:
            return None
        dev = self.potential_grid.device
        if self.density_grid.device != dev:
            raise ValueError("potential_grid y density_grid están en devices distintos")
        return jnp.stack([self.potential_grid, self.density_grid], axis=-1)
    
    def to_device(self, device, include_grids: bool = True):
        pos = jax.device_put(self.positions, device)
        vel = jax.device_put(self.velocities, device)
        pot = jax.device_put(self.potential, device)

        # Solo movemos al GPU si se solicita Y si existen los datos
        potg = None
        deng = None
        if include_grids:
            if self.potential_grid is not None:
                potg = jax.device_put(self.potential_grid, device)
            if self.density_grid is not None:
                deng = jax.device_put(self.density_grid, device)

        return ResolutionData(
            mesh=self.mesh,
            positions=pos,
            velocities=vel,
            potential=pot,
            potential_grid=potg,
            density_grid=deng,
        )
    # def to_device(self, device):
    #     """
    #     Return a NEW ResolutionData on `device` (does NOT mutate the original object).
    #     TRANSPORT ONLY: NO RESCALING HERE (prevents double scaling / mesh^2 bugs).
    #     """
    #     pos = jax.device_put(self.positions, device)
    #     vel = jax.device_put(self.velocities, device)
    #     pot = jax.device_put(self.potential, device)

    #     potg = jax.device_put(self.potential_grid, device) if self.potential_grid is not None else None
    #     deng = jax.device_put(self.density_grid, device) if self.density_grid is not None else None

    #     return ResolutionData(
    #         mesh=self.mesh,
    #         positions=pos,
    #         velocities=vel,
    #         potential=pot,
    #         potential_grid=potg,
    #         density_grid=deng,
    #         grid=None,
    #         _scaled_to_mesh=False,
    #     )

    # PyTree methods (no derived grid)
    def tree_flatten(self):
        children = (
            self.positions,
            self.velocities,
            self.potential,
            self.potential_grid,
            self.density_grid,
        )
        aux = {"mesh": self.mesh, "_scaled_to_mesh": self._scaled_to_mesh}
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        positions, velocities, potential, potential_grid, density_grid = children
        obj = cls(
            mesh=aux["mesh"],
            positions=positions,
            velocities=velocities,
            potential=potential,
            potential_grid=potential_grid,
            density_grid=density_grid,
        )
        obj._scaled_to_mesh = aux.get("_scaled_to_mesh", False)
        obj.grid = None
        return obj
    
class PMDataset:
    def __init__(self, high_res_data, low_res_data, infinite=False, need_grid=False):
        self.hr = high_res_data
        self.lr = low_res_data
        self.infinite = infinite
        self.need_grid = need_grid
        self.iterator = iter(self)

    def __len__(self):
        return len(self.hr)

    def move_to_device(self, batch_data, device, *, build_grid: Optional[bool] = None):
        if build_grid is None:
            build_grid = self.need_grid

        # Pasamos build_grid para que to_device decida si sube los datos al GPU
        hr = batch_data["hr"].to_device(device, include_grids=False) # HR usualmente no necesita grid
        lr = batch_data["lr"].to_device(device, include_grids=build_grid)

        if build_grid:
            lr.grid = jnp.stack([lr.potential_grid, lr.density_grid], axis=-1)
        else:
            lr.grid = None

        return {"hr": hr, "lr": lr}

    def __getitem__(self, idx):
        high_res_data = self.hr[idx]
        low_res_data = self.lr[idx]
        return {"hr": high_res_data, "lr": low_res_data}

    def __iter__(self):
        if self.infinite:
            while True:
                for i in range(len(self)):
                    yield self[i]
        else:
            for i in range(len(self)):
                yield self[i]

def load_dataset_for_sim_idx_list(
    idx_list,
    mesh_hr,
    mesh_lr,
    data_dir,
    box_size,
    snapshots=None,
    move_to_cpu=True,
    need_grid=False,
):
    grid_factor = mesh_hr / mesh_lr
    low_res_data, high_res_data = [], []

    # factor de conversión: de [0,1) (box-normalized) -> [0, mesh_lr) (mesh_lr units)
    m_lr = jnp.asarray(mesh_lr, dtype=jnp.float32)

    for idx in idx_list:
        logger.info(
            f"Loading data for simulation index {idx} with mesh_hr={mesh_hr}, mesh_lr={mesh_lr}, "
            f"box_size={box_size}, snapshots={snapshots}"
        )

        # HR particles (pos/vel) vienen normalizados a box si normalize_to_box=True en get_data
        pos_hr, vel_hr, grav_pot_hr = get_data(
            data_dir=data_dir,
            n_mesh=mesh_hr,
            downsampling_factor=None,
            idx=idx,
            box_size=box_size,
            snapshots=snapshots,
            move_to_cpu=move_to_cpu,
            normalize_to_box=True,
        )

        # LR particles + grids
        pos_lr, vel_lr, grav_pot_lr, grav_pot_grid_lr, dens_grid_lr = get_data(
            data_dir=data_dir,
            n_mesh=mesh_lr,
            get_grids=True,
            idx=idx,
            box_size=box_size,
            snapshots=snapshots,
            move_to_cpu=move_to_cpu,
            normalize_to_box=True,
        )

        # -------------------------------
        # 1) Asegura float32 (VRAM + estabilidad)
        # -------------------------------
        pos_hr = pos_hr.astype(jnp.float32)
        vel_hr = vel_hr.astype(jnp.float32)
        pos_lr = pos_lr.astype(jnp.float32)
        vel_lr = vel_lr.astype(jnp.float32)

        grav_pot_hr = grav_pot_hr.astype(jnp.float32)
        grav_pot_lr = grav_pot_lr.astype(jnp.float32)
        grav_pot_grid_lr = grav_pot_grid_lr.astype(jnp.float32)
        dens_grid_lr = dens_grid_lr.astype(jnp.float32)

        # -------------------------------
        # 2) Convertir posiciones/velocidades a UNIDADES mesh_lr
        #    (esto evita hacer "* mesh" dentro de la loss y garantiza consistencia LR/HR)
        # -------------------------------
        pos_hr = pos_hr * m_lr
        vel_hr = vel_hr * m_lr
        pos_lr = pos_lr * m_lr
        vel_lr = vel_lr * m_lr

        # -------------------------------
        # 3) Escalamiento de grids/potential LR (tu lógica original)
        # -------------------------------
        particle_factor = len(pos_hr[0]) / len(pos_lr[0])
        up_resolution_factor = particle_factor / grid_factor

        grav_pot_grid_lr = grav_pot_grid_lr * up_resolution_factor
        dens_grid_lr = dens_grid_lr * up_resolution_factor
        grav_pot_lr = grav_pot_lr * up_resolution_factor

        # -------------------------------
        # 4) IMPORTANTE: ahora TODO está en unidades mesh_lr
        #    Así que el "mesh" que debe usarse por la dinámica/loss es mesh_lr (también para HR)
        # -------------------------------
        high_res_data.append(
            ResolutionData(mesh_lr, pos_hr, vel_hr, grav_pot_hr, None, None)
        )
        low_res_data.append(
            ResolutionData(
                mesh_lr, pos_lr, vel_lr, grav_pot_lr, grav_pot_grid_lr, dens_grid_lr
            )
        )

    return low_res_data, high_res_data


def load_datasets(
    n_train_sims,
    n_val_sims,
    n_test_sims,
    mesh_hr,
    mesh_lr,
    data_dir,
    box_size,
    snapshots=None,
    need_grid=False,
):
    logger.info(f"Loading datasets with n_train_sims={n_train_sims}, n_val_sims={n_val_sims}, n_test_sims={n_test_sims}, mesh_hr={mesh_hr}, mesh_lr={mesh_lr}, box_size={box_size}, snapshots={snapshots}")
    logger.info(f"Including grid {need_grid}")
    val_idx_list = list(range(n_val_sims))
    train_idx_list = list(range(n_val_sims, n_val_sims + n_train_sims))
    test_idx_list = list(
        range(
            n_val_sims + n_train_sims,
            n_val_sims + n_train_sims + n_test_sims,
        )
    )
    train_low_res_data, train_high_res_data = load_dataset_for_sim_idx_list(
        train_idx_list,
        mesh_hr,
        mesh_lr,
        data_dir,
        box_size=box_size,
        snapshots=snapshots,
        need_grid=need_grid,
    )
    val_low_res_data, val_high_res_data = load_dataset_for_sim_idx_list(
        val_idx_list,
        mesh_hr,
        mesh_lr,
        data_dir,
        box_size=box_size,
        snapshots=snapshots,
        need_grid=need_grid,
    )
    test_low_res_data, test_high_res_data = load_dataset_for_sim_idx_list(
        test_idx_list,
        mesh_hr,
        mesh_lr,
        data_dir,
        box_size=box_size,
        snapshots=snapshots,
        need_grid=need_grid,
    )
    return (
        PMDataset(
            train_high_res_data,
            train_low_res_data,
            infinite=True,
            need_grid=need_grid,
        ),
        PMDataset(val_high_res_data, val_low_res_data,need_grid=need_grid,),
        PMDataset(test_high_res_data, test_low_res_data,need_grid=need_grid,),
    )
