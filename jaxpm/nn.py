import jax
import jax.numpy as jnp
import haiku as hk

from jaxpm.painting import cic_read
from typing import Optional


def _deBoorVectorized(x, t, c, p):
    """
    Evaluates S(x).

    Args
    ----
    x: position
    t: array of knot positions, needs to be padded as described above
    c: array of control points
    p: degree of B-spline
    """
    k = jnp.digitize(x, t) - 1

    d = [c[j + k - p] for j in range(0, p + 1)]
    for r in range(1, p + 1):
        for j in range(p, r - 1, -1):
            alpha = (x - t[j + k - p]) / (t[j + 1 + k - r] - t[j + k - p])
            d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j]
    return d[p]


class NeuralSplineFourierFilter(hk.Module):
    """A rotationally invariant filter parameterized by
    a b-spline with parameters specified by a small NN."""

    def __init__(self, n_knots=8, latent_size=16, name=None):
        """
        n_knots: number of control points for the spline
        """
        super().__init__(name=name)
        self.n_knots = n_knots
        self.latent_size = latent_size

    def __call__(self, x, a):
        """
        x: array, scale, normalized to fftfreq default
        a: scalar, scale factor
        """

        net = jnp.sin(hk.Linear(self.latent_size)(jnp.atleast_1d(a)))
        net = jnp.sin(hk.Linear(self.latent_size)(net))

        w = hk.Linear(self.n_knots + 1)(net)
        k = hk.Linear(self.n_knots - 1)(net)

        # make sure the knots sum to 1 and are in the interval 0,1
        k = jnp.concatenate([jnp.zeros((1,)), jnp.cumsum(jax.nn.softmax(k))])

        w = jnp.concatenate([jnp.zeros((1,)), w])

        # Augment with repeating points
        ak = jnp.concatenate([jnp.zeros((3,)), k, jnp.ones((3,))])

        return _deBoorVectorized(jnp.clip(x / jnp.sqrt(3), 0, 1 - 1e-4), ak, w, 3)


class Rescale(hk.Module):
    def __init__(self, n_input):
        super().__init__()
        self.scale = hk.get_parameter(
            "scale", [n_input], init=jnp.ones, dtype=jnp.float64
        )
        self.bias = hk.get_parameter(
            "bias", [n_input], init=jnp.zeros, dtype=jnp.float64
        )

    def __call__(self, x):
        return self.scale * x + self.bias


class ConvBlock(hk.Module):
    def __init__(
        self,
        output_channels: int,
        kernel_size: int = 3,
        padding: str = "SAME",
        activation=jax.nn.relu,
        pad_periodic: bool = False,
        global_conditioning: Optional[str] = None,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.pad_periodic = pad_periodic
        self.conv = hk.Conv3D(
            output_channels=output_channels,
            kernel_shape=(
                kernel_size,
                kernel_size,
                kernel_size,
            ),
            padding=padding,
        )
        self.activation = activation
        self.global_conditioning = global_conditioning
        if self.global_conditioning is not None and self.global_conditioning == "add":
            # Need to make sure that the number of channels is the same as the number of global features
            # since we will add them up
            self.globals_fcn = hk.nets.MLP([output_channels] * 2)

    def add_global_conditioning(self, x, global_features):
        if self.global_conditioning == "add":
            return x + global_features[:, None, None, :]
        elif self.global_conditioning == "concat":
            tiled_global_features = jnp.tile(
                global_features, (x.shape[0], x.shape[1], x.shape[2], 1)
            )
            return jnp.concatenate([x, tiled_global_features], axis=-1)
        else:
            raise NotImplementedError(
                f"Global conditioning {self.global_conditioning} not implemented"
            )

    def __call__(
        self,
        x,
    ):
        x, global_features = x
        if self.pad_periodic:
            x = jnp.pad(
                x,
                pad_width=(
                    (self.kernel_size, self.kernel_size),
                    (self.kernel_size, self.kernel_size),
                    (self.kernel_size, self.kernel_size),
                    (0, 0),
                ),
                mode="wrap",
            )
        if self.global_conditioning is not None and global_features is not None:
            x = self.add_global_conditioning(
                x,
                global_features,
            )
        x = self.conv(x)
        x = self.activation(x)
        if self.pad_periodic:
            x = x[
                ...,
                self.kernel_size : -self.kernel_size,
                self.kernel_size : -self.kernel_size,
                self.kernel_size : -self.kernel_size,
                :,
            ]
        if self.global_conditioning is not None and self.global_conditioning == "add":
            global_features = self.globals_fcn(
                global_features,
            )
        return (x, global_features)


class FullyConnectedBlock(hk.Module):
    def __init__(
        self,
        hidden_dim: int,
        activation=jax.nn.relu,
    ):
        super().__init__()
        self.fc = hk.Linear(hidden_dim)
        self.activation = activation

    def __call__(self, x):
        return self.activation(self.fc(x))


class AttentionRead(hk.Module):
    def __init__(
        self,
        attn_hidden_dim=16,
        attn_scores_dim=16,
        n_fully_connected=3,
    ):
        super().__init__()
        self.mlp_distances = hk.nets.MLP(
            output_sizes=[attn_hidden_dim] * n_fully_connected
            + [
                attn_scores_dim,
            ],
        )
        self.mlp_mesh = hk.nets.MLP(
            output_sizes=[attn_hidden_dim] * n_fully_connected
            + [
                attn_scores_dim,
            ],
        )

    def compute_attention(self, query, key):
        return jax.nn.softmax(jnp.sum(query * key, axis=-1))

    def __call__(
        self,
        mesh,
        positions,
    ):
        positions = jnp.expand_dims(positions, 1)
        floor = jnp.floor(positions)
        connection = jnp.array(
            [
                [
                    [0, 0, 0],
                    [1.0, 0, 0],
                    [0.0, 1, 0],
                    [0.0, 0, 1],
                    [1.0, 1, 0],
                    [1.0, 0, 1],
                    [0.0, 1, 1],
                    [1.0, 1, 1],
                ]
            ]
        )
        neighbour_coords = floor + connection
        distances = positions - neighbour_coords
        neighbour_coords = jnp.mod(
            neighbour_coords.astype("int32"), jnp.array(mesh.shape)
        )
        neighbour_mesh = mesh[
            neighbour_coords[..., 0],
            neighbour_coords[..., 1],
            neighbour_coords[..., 3],
        ]
        query = jax.vmap(self.mlp_distances)(distances)
        key = jax.vmap(self.mlp_mesh)(neighbour_mesh[..., None])
        # Using vmap for compute_attention and dot product
        attention_weights = jax.vmap(self.compute_attention)(query, key)
        return jax.vmap(jnp.dot)(attention_weights, neighbour_mesh)


class CNN(hk.Module):
    def __init__(
        self,
        channels_hidden_dim: int,
        n_convolutions: int,
        n_fully_connected: int,
        n_globals_embedding: int = 2,
        globals_embedding_dim: int = 8,
        input_dim: int = 2,
        output_dim: int = 1,
        kernel_size: int = 3,
        pad_periodic: bool = True,
        embed_globals: bool = True,
        global_conditioning: Optional[str] = None,
        use_attention_interpolation: bool = False,
        add_particle_velocities: bool = False,
    ):
        super().__init__(name="CNN")
        self.kernel_size = kernel_size
        self.n_convolutions = n_convolutions
        self.pad_periodic = pad_periodic
        self.embed_globals = embed_globals
        self.learned_norm = Rescale(input_dim)
        self.conv_block = hk.Sequential(
            [
                ConvBlock(
                    output_channels=channels_hidden_dim,
                    kernel_size=kernel_size,
                    pad_periodic=pad_periodic,
                    global_conditioning=global_conditioning,
                )
                for _ in range(n_convolutions)
            ]
        )
        self.use_attention_interpolation = use_attention_interpolation
        if use_attention_interpolation:
            attention_read = AttentionRead()
            self.read_features_at_pos = jax.vmap(
                attention_read,
                in_axes=(-1, None),
            )
        else:
            self.read_features_at_pos = jax.vmap(
                cic_read,
                in_axes=(-1, None),
            )
        self.fcn_block = hk.nets.MLP(
            output_sizes=[channels_hidden_dim] * n_fully_connected
            + [
                output_dim,
            ],
        )
        if self.embed_globals:
            self.globals_fcn = hk.nets.MLP(
                output_sizes=[globals_embedding_dim] * n_globals_embedding,
            )
        self.add_particle_velocities = add_particle_velocities

    def concatenate_globals(self, features_at_pos, global_features):
        broadcast_globals = jnp.broadcast_to(
            global_features,
            (
                features_at_pos.shape[0],
                global_features.shape[-1],
            ),
        )
        return jnp.concatenate([features_at_pos, broadcast_globals], axis=-1)

    def get_feature_maps(self, x, global_features=None):
        if global_features is not None:
            if global_features.ndim < 2:
                global_features = jnp.expand_dims(global_features, axis=0)
                if global_features.ndim < 2:
                    global_features = jnp.expand_dims(global_features, axis=0)
            if self.embed_globals:
                global_features = self.globals_fcn(global_features)
        x = self.learned_norm(x)  # [LR, LR, LR, input_dim]
        x, global_features = self.conv_block(
            (x, global_features)
        )  # [LR, LR, LR, n_channels_hidden]
        return x, global_features

    def __call__(
        self,
        x,
        positions,
        global_features=None,
        return_features=False,
        velocities=None,
    ):
        x, global_features = self.get_feature_maps(
            x=x,
            global_features=global_features,
        )
        if positions.ndim == 1:
            positions = positions[None, ...]
        # swap axes to make the last axis the feature axis for the linear layers
        features_at_pos = self.read_features_at_pos(x, positions).swapaxes(-2, -1)
        # [n_particles, n_channels_hidden]
        # Add time as a feature
        if global_features is not None:
            features_at_pos = self.concatenate_globals(
                features_at_pos,
                global_features,
            ).squeeze()
            # [n_particles, n_channels_hidden + 1]
        if self.add_particle_velocities:
            features_at_pos = jnp.concatenate([features_at_pos, velocities], axis=-1)
        features_at_pos = self.fcn_block(features_at_pos)
        # [n_particles, output_dim]
        if return_features:
            return features_at_pos, x
        return features_at_pos
