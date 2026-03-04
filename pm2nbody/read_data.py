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


# def get_data(
#     data_dir: Path,
#     n_mesh: int,
#     downsampling_factor: Optional[int] = None,
#     get_grids: Optional[bool] = False,
#     snapshots: Optional[List[int]] = None,
#     box_size: Optional[float] = 256.0,
#     normalize_to_box: Optional[bool] = True,
#     idx: Optional[int] = 0,
#     move_to_cpu: bool = True,
# ):
#     pos = jnp.load(data_dir / f"pos_m{n_mesh}_s{idx}.npy")
#     if snapshots is None:
#         snapshots = jnp.arange(len(pos))
#     pos = pos[snapshots, :, :]
#     vel = jnp.load(data_dir / f"vel_m{n_mesh}_s{idx}.npy")[snapshots]
#     if normalize_to_box:
#         pos /= box_size
#         vel /= box_size
#     gravitational_potential = jnp.load(data_dir / f"pot_m{n_mesh}_s{idx}.npy")[
#         snapshots
#     ]
#     if downsampling_factor is not None:
#         pos = downsample_to_mesh(
#             array=pos,
#             n_mesh=n_mesh,
#             downsampling_factor=downsampling_factor,
#         )
#         vel = downsample_to_mesh(
#             array=vel,
#             n_mesh=n_mesh,
#             downsampling_factor=downsampling_factor,
#         )
#         gravitational_potential = downsample_to_mesh(
#             array=gravitational_potential,
#             n_mesh=n_mesh,
#             downsampling_factor=downsampling_factor,
#         )
#     if not get_grids:
#         return pos, vel, gravitational_potential
#     potential_grid = jnp.load(data_dir / f"pot_grid_m{n_mesh}_s{idx}.npy")[snapshots]
#     density_grid = jnp.load(data_dir / f"dens_grid_m{n_mesh}_s{idx}.npy")[snapshots]
#     # Move all data to cpu:
#     if move_to_cpu:
#         cpu = jax.devices("cpu")[0]
#         pos = jax.device_put(pos, cpu)
#         vel = jax.device_put(vel, cpu)
#         gravitational_potential = jax.device_put(gravitational_potential, cpu)
#         potential_grid = jax.device_put(potential_grid, cpu)
#         density_grid = jax.device_put(density_grid, cpu)
#     return pos, vel, gravitational_potential, potential_grid, density_grid

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
        gravitational_potential = jnp.asarray(gravitational_potential)

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

# @dataclass
# class ResolutionData:
#     mesh: int
#     positions: jnp.array
#     velocities: jnp.array
#     potential: jnp.array
#     potential_grid: Optional[jnp.array] = None
#     density_grid: Optional[jnp.array] = None
#     grid: Optional[jnp.array] = None

#     def __post_init__(
#         self,
#     ):
#         if self.potential_grid is not None and self.density_grid is not None:
#             self.grid = jnp.stack([self.potential_grid, self.density_grid], axis=-1)

#     def move_to_device(
#         self,
#         device,
#     ):
#         self.positions = jax.device_put(self.positions, device)
#         self.velocities = jax.device_put(self.velocities, device)
#         self.mesh = jax.device_put(self.mesh, device)

# 1. Add the PyTree decorator ABOVE the dataclass decorator
@jax.tree_util.register_pytree_node_class
@dataclass
class ResolutionData:
    mesh: int
    positions: jnp.ndarray
    velocities: jnp.ndarray
    potential: jnp.ndarray
    potential_grid: Optional[jnp.ndarray] = None
    density_grid: Optional[jnp.ndarray] = None
    grid: Optional[jnp.ndarray] = None

    def __post_init__(self):
        # we do not build grid here
        self.grid = None

    def get_grid(self):
        """Devuelve grid en el mismo device que potential_grid/density_grid.
        Re-construye solo si no existe o si cambió el device."""
        if self.potential_grid is None or self.density_grid is None:
            return None

        # Ambos deben estar en el mismo device
        dev = self.potential_grid.device
        if self.density_grid.device != dev:
            raise ValueError("potential_grid y density_grid están en devices distintos")
        if self.grid is not None and getattr(self, "_grid_device", None) == dev:
            return self.grid

        # Construye y cachea
        self.grid = jnp.stack([self.potential_grid, self.density_grid], axis=-1)
        self._grid_device = dev
        return self.grid
    
    def move_to_device(self, device):
        # Particle fields
        self.positions = jax.device_put(self.positions, device)
        self.velocities = jax.device_put(self.velocities, device)
        self.potential = jax.device_put(self.potential, device)

        # Grid fields (if present)
        if self.potential_grid is not None:
            self.potential_grid = jax.device_put(self.potential_grid, device)
        if self.density_grid is not None:
            self.density_grid = jax.device_put(self.density_grid, device)

        # Build grid on the target device (now guaranteed same device)
        if self.potential_grid is not None and self.density_grid is not None:
            self.grid = jnp.stack([self.potential_grid, self.density_grid], axis=-1)
            self._grid_device = device # Keep tracker in sync
        else:
            self.grid = None

        # Keep mesh as Python int (do NOT device_put mesh)
        return self

    # ==========================================
    # JAX PyTree Registration Methods
    # ==========================================

    def tree_flatten(self):
        """Tells JAX which attributes are arrays to trace, and which are static metadata."""
        # ALL arrays must go into children (even if they are None)
        children = (
            self.positions, 
            self.velocities, 
            self.potential, 
            self.potential_grid, 
            self.density_grid, 
            self.grid
        )
        
        # Static variables (ints, strings, etc.) go into aux_data
        aux_data = {
            "mesh": self.mesh
        }
        
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Tells JAX how to reconstruct the object inside the JIT compiler."""
        # 1. Unpack arrays in the exact order they were packed
        positions, velocities, potential, potential_grid, density_grid, grid = children
        
        # 2. Re-instantiate the dataclass. 
        # WARNING: This will trigger __post_init__(), which sets obj.grid = None
        obj = cls(
            mesh=aux_data["mesh"],
            positions=positions,
            velocities=velocities,
            potential=potential,
            potential_grid=potential_grid,
            density_grid=density_grid,
        )
        
        # 3. CRITICAL: Override __post_init__ to restore the traced JAX array for grid!
        obj.grid = grid
        
        return obj

class PMDataset:
    def __init__(
        self,
        high_res_data,
        low_res_data,
        infinite=False,
    ):
        self.hr = high_res_data
        self.lr = low_res_data
        self.infinite = infinite
        self.iterator = iter(self)

    def __len__(
        self,
    ):
        return len(self.hr)

    def move_to_device(self, batch_data, device):
        batch_data["hr"].move_to_device(device)
        batch_data["lr"].move_to_device(device)
        return {
            "hr": batch_data["hr"],
            "lr": batch_data["lr"],
        }

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
):
    grid_factor = mesh_hr / mesh_lr
    low_res_data, high_res_data = [], []
    for idx in idx_list:
        logger.info(f"Loading data for simulation index {idx} with mesh_hr={mesh_hr}, mesh_lr={mesh_lr}, box_size={box_size}, snapshots={snapshots}")
        pos_hr, vel_hr, grav_pot_hr = get_data(
            data_dir=data_dir,
            n_mesh=mesh_hr,
            downsampling_factor=None,  # mesh_hr // mesh_lr,
            idx=idx,
            box_size=box_size,
            snapshots=snapshots,
            move_to_cpu=move_to_cpu,
        )
        pos_lr, vel_lr, grav_pot_lr, grav_pot_grid_lr, dens_grid_lr = get_data(
            data_dir=data_dir,
            n_mesh=mesh_lr,
            get_grids=True,
            idx=idx,
            box_size=box_size,
            snapshots=snapshots,
            move_to_cpu=move_to_cpu,
        )
        particle_factor = len(pos_hr[0]) / len(pos_lr[0])
        up_resolution_factor = particle_factor / grid_factor
        grav_pot_grid_lr *= up_resolution_factor
        dens_grid_lr *= up_resolution_factor
        grav_pot_lr *= up_resolution_factor
        high_res_data.append(
            ResolutionData(mesh_hr, pos_hr, vel_hr, grav_pot_hr, None, None)
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
):
    logger.info(f"Loading datasets with n_train_sims={n_train_sims}, n_val_sims={n_val_sims}, n_test_sims={n_test_sims}, mesh_hr={mesh_hr}, mesh_lr={mesh_lr}, box_size={box_size}, snapshots={snapshots}")
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
    )
    val_low_res_data, val_high_res_data = load_dataset_for_sim_idx_list(
        val_idx_list,
        mesh_hr,
        mesh_lr,
        data_dir,
        box_size=box_size,
        snapshots=snapshots,
    )
    test_low_res_data, test_high_res_data = load_dataset_for_sim_idx_list(
        test_idx_list,
        mesh_hr,
        mesh_lr,
        data_dir,
        box_size=box_size,
        snapshots=snapshots,
    )
    return (
        PMDataset(
            train_high_res_data,
            train_low_res_data,
            infinite=True,
        ),
        PMDataset(val_high_res_data, val_low_res_data),
        PMDataset(test_high_res_data, test_low_res_data),
    )
