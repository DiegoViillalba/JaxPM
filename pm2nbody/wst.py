# ==============================================================================
# wst.py — Wavelet Scattering Transform 3D como capa diferenciable en JAX/Haiku
#
# Implementa WST de primer orden para campos cosmológicos 3D.
# Diseñado para integrarse como preprocesador del CNN en jaxpm/nn.py.
#
# Referencia teórica:
#   Mallat 2012, "Group Invariant Scattering"
#   Allys et al. 2020, "New interpretable statistics for large-scale structure"
#   Regaldo-Saint Blancard et al. 2022, "A new approach to observational cosmology"
#
# Pipeline:
#   x [N,N,N] → Morlet wavelets en J escalas × L orientaciones
#             → |W_jl * x| → promedio espacial
#             → coeficientes S1 [J*L, N, N, N] (campo local)
#             → concatenar con x → [J*L+1, N, N, N] → CNN
#
# Por qué WST y no solo más canales CNN:
#   El P(k) captura correlaciones de orden 2. Los halos cosmológicos tienen
#   correlaciones de orden superior (filamentos, vacíos) que el CNN tarda
#   muchas capas en aprender desde el potencial crudo. La WST pre-computa
#   estas correlaciones de forma analítica y diferenciable.
# ==============================================================================

import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
from typing import Optional


# ==============================================================================
# Construcción de filtros Morlet 3D
# ==============================================================================

def _morlet_3d_fourier(shape, j, theta, phi, sigma_spatial=0.8, xi=3 * np.pi / 4):
    """
    Construye un filtro Morlet 3D en espacio de Fourier.

    El wavelet de Morlet es una gaussiana modulada por una exponencial compleja.
    En Fourier es una gaussiana centrada en la frecuencia de la orientación.

    Parámetros
    ----------
    shape : tuple (N, N, N)
        Forma del campo de entrada.
    j : int
        Índice de escala. La frecuencia central es xi * 2^(-j).
    theta : float
        Ángulo polar en radianes [0, π].
    phi : float
        Ángulo azimutal en radianes [0, 2π].
    sigma_spatial : float
        Ancho de la gaussiana en espacio real (relativo a la escala 2^j).
    xi : float
        Frecuencia central del wavelet madre.

    Retorna
    -------
    psi_f : jnp.ndarray [N, N, N], complejo
        Filtro en espacio de Fourier, listo para multiplicar con FFT(x).
    """
    N = shape[0]

    # Frecuencias en cada dirección — fftfreq normalizado a [-π, π]
    kx = jnp.fft.fftfreq(N) * 2 * np.pi
    ky = jnp.fft.fftfreq(N) * 2 * np.pi
    kz = jnp.fft.fftfreq(N) * 2 * np.pi
    KX, KY, KZ = jnp.meshgrid(kx, ky, kz, indexing='ij')

    # Vector de orientación 3D (esférico → cartesiano)
    ux = np.sin(theta) * np.cos(phi)
    uy = np.sin(theta) * np.sin(phi)
    uz = np.cos(theta)

    # Frecuencia central del wavelet a escala j
    # A escala j, el wavelet cubre frecuencias ~xi * 2^(-j)
    scale  = 2.0 ** j
    xi_j   = xi / scale

    # Centro del filtro gaussiano en Fourier
    k0x = xi_j * ux
    k0y = xi_j * uy
    k0z = xi_j * uz

    # Ancho de la gaussiana en Fourier (inversamente proporcional a escala)
    sigma_f = 1.0 / (sigma_spatial * scale)

    # Gaussiana centrada en (k0x, k0y, k0z)
    psi_f = jnp.exp(
        -0.5 * (
            (KX - k0x) ** 2 +
            (KY - k0y) ** 2 +
            (KZ - k0z) ** 2
        ) / sigma_f ** 2
    )

    # Normalización L2
    norm = jnp.sqrt(jnp.sum(psi_f ** 2)) + 1e-8
    return psi_f / norm


def build_morlet_filters(shape, J, L):
    """Genera exactamente J*L filtros — J escalas × L orientaciones uniformes."""
    filters      = []
    orientations = []

    # L orientaciones distribuidas uniformemente en el hemisferio norte de S²
    # Usamos coordenadas de Fibonacci para distribución casi uniforme
    for l in range(L):
        # Ángulo polar: L niveles uniformes entre 0 y π/2
        theta = np.pi * (l + 0.5) / L
        # Ángulo azimutal: L pasos uniformes en 2π
        phi   = 2 * np.pi * l / L
        orientations.append((theta, phi))

    for j in range(J):
        for theta, phi in orientations:
            psi = _morlet_3d_fourier(shape, j + 1, theta, phi)
            filters.append(psi)

    assert len(filters) == J * L, f"Se esperaban {J*L} filtros, se generaron {len(filters)}"
    return filters, orientations * J

# ==============================================================================
# Capa WST diferenciable en Haiku
# ==============================================================================

class WaveletScatteringTransform(hk.Module):
    """
    Capa WST de primer orden como módulo Haiku diferenciable.

    Aplica J*L filtros Morlet al campo de entrada, toma el módulo
    y devuelve los coeficientes de scattering como canales adicionales.

    Los filtros son parámetros FIJOS (no entrenables) — son funciones
    matemáticas deterministas. Solo el CNN posterior es entrenable.

    Uso en CNN:
        x [N,N,N,1] → WST → [N,N,N, J*L+1] → Conv3D ...

    Parámetros
    ----------
    J : int
        Número de escalas. J=2 recomendado para empezar (bajo costo).
    L : int
        Número de orientaciones. L=4 recomendado (equilibrio costo/calidad).
    normalize : bool
        Si True, normaliza cada canal de scattering por su media espacial.
        Recomendado para estabilidad del entrenamiento.
    """

    def __init__(self, J: int = 2, L: int = 4,
                 normalize: bool = True,
                 name: Optional[str] = None):
        super().__init__(name=name or "WST")
        self.J         = J
        self.L         = L
        self.normalize = normalize
        self.n_filters = J * L

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Parámetros
        ----------
        x : jnp.ndarray [N, N, N, C]
            Campo de entrada con C canales. Típicamente C=1 (potencial).

        Retorna
        -------
        out : jnp.ndarray [N, N, N, C + J*L*C]
            Campo original concatenado con coeficientes de scattering.
            Cada canal de entrada produce J*L canales de scattering.
        """
        N      = x.shape[0]
        shape  = (N, N, N)

        # Construir banco de filtros (fijo, no entrenable)
        filters, _ = build_morlet_filters(shape, self.J, self.L)

        # Aplicar WST SOLO al primer canal (potencial gravitacional, canal 0).
        # Si x tiene C=2 (potencial + densidad), el segundo canal se pasa
        # sin modificar. Esto garantiza que la salida siempre tiene
        # C + J*L canales independientemente de C.
        #
        # Con C=1: salida [N,N,N, 1 + J*L]
        # Con C=2: salida [N,N,N, 2 + J*L]
        # → input_dim del CNN debe ser C_entrada + J*L
        xc   = x[..., 0]          # [N, N, N] — potencial gravitacional
        xc_f = jnp.fft.fftn(xc)   # FFT 3D

        scat_channels = []
        for psi_f in filters:
            wxc_f = xc_f * psi_f               # convolución en Fourier
            wxc   = jnp.fft.ifftn(wxc_f)       # volver a espacio real
            s1    = jnp.abs(wxc)               # módulo — invariante a fase

            if self.normalize:
                s1 = s1 / (jnp.mean(s1) + 1e-8)

            scat_channels.append(s1)

        # Apilar J*L coeficientes: [N, N, N, J*L]
        scat = jnp.stack(scat_channels, axis=-1)

        # Concatenar todos los canales originales + scattering: [N,N,N, C+J*L]
        return jnp.concatenate([x, scat], axis=-1)

    @property
    def output_channels(self) -> int:
        """Canales de salida para input con C=1. Para C>1 usar C + J*L."""
        return 1 + self.n_filters


# ==============================================================================
# Integración con CNN existente — modificación mínima de nn.py
# ==============================================================================

class CNNWithWST(hk.Module):
    """
    CNN con WST preprocesador.

    Wrapper que antepone la capa WST al CNN existente.
    Solo cambia input_dim — el resto del CNN es idéntico.

    Uso en train_refactored.py:
        Reemplazar build_network para type="cnn_wst"
    """

    def __init__(
        self,
        J: int = 2,
        L: int = 4,
        channels_hidden_dim: int = 16,
        n_convolutions: int = 3,
        n_fully_connected: int = 2,
        kernel_size: int = 3,
        pad_periodic: bool = True,
        embed_globals: bool = False,
        n_globals_embedding: int = 1,
        globals_embedding_dim: int = 64,
        global_conditioning: str = "add",
        use_attention_interpolation: bool = False,
        add_particle_velocities: bool = True,
        output_dim: int = 1,
    ):
        super().__init__(name="CNNWithWST")

        # Guardar todos los parámetros — los módulos Haiku se instancian
        # en __call__ para que el name scope sea correcto
        self.J                        = J
        self.L                        = L
        # wst_input_dim se calcula dinámicamente en __call__ según C real
        # Para C=1 (mse_positions): 1 + J*L
        # Para C=2 (need_grid=True): 2 + J*L
        # Se pasa como argumento al CNN en __call__
        self.J_times_L                = J * L
        self.channels_hidden_dim      = channels_hidden_dim
        self.n_convolutions           = n_convolutions
        self.n_fully_connected        = n_fully_connected
        self.kernel_size              = kernel_size
        self.pad_periodic             = pad_periodic
        self.embed_globals            = embed_globals
        self.n_globals_embedding      = n_globals_embedding
        self.globals_embedding_dim    = globals_embedding_dim
        self.global_conditioning      = global_conditioning
        self.use_attention_interpolation = use_attention_interpolation
        self.add_particle_velocities  = add_particle_velocities
        self.output_dim               = output_dim

    def __call__(self, x, positions, global_features=None,
                 return_features=False, velocities=None):
        """
        x : [N, N, N, 1] — campo de entrada (potencial o densidad)
        Aplica WST → [N, N, N, 1+J*L] → CNN → corrección [N_part, output_dim]
        """
        from jaxpm.nn import CNN

        # WST: enriquece el campo con coeficientes multi-escala
        wst   = WaveletScatteringTransform(
            J=self.J, L=self.L, normalize=True, name="wst"
        )
        x_wst = wst(x)    # [N, N, N, C + J*L]

        # input_dim real = canales originales C + J*L filtros WST
        # Calculado dinámicamente para soportar C=1 o C=2
        actual_input_dim = x.shape[-1] + self.J_times_L

        # CNN con input_dim correcto
        cnn = CNN(
            channels_hidden_dim=self.channels_hidden_dim,
            n_convolutions=self.n_convolutions,
            n_fully_connected=self.n_fully_connected,
            input_dim=actual_input_dim,
            output_dim=self.output_dim,
            kernel_size=self.kernel_size,
            pad_periodic=self.pad_periodic,
            embed_globals=self.embed_globals,
            n_globals_embedding=self.n_globals_embedding,
            globals_embedding_dim=self.globals_embedding_dim,
            global_conditioning=self.global_conditioning,
            use_attention_interpolation=self.use_attention_interpolation,
            add_particle_velocities=self.add_particle_velocities,
        )

        return cnn(
            x=x_wst,
            positions=positions,
            global_features=global_features,
            return_features=return_features,
            velocities=velocities,
        )