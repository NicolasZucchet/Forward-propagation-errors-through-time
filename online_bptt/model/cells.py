import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any
from functools import partial
import warnings

# Remove all complex values warnings (appears when taking real value)
warnings.filterwarnings("ignore", category=jnp.ComplexWarning)


def chrono_init(key, shape, dtype=jnp.float32, T_min=1.0, T_max=10.0):
    if T_min is None or T_max is None:
        return jnp.zeros(shape, dtype=dtype)
    return jnp.log(jax.random.uniform(key, shape, dtype=dtype, minval=T_min, maxval=T_max))


def nu_init(key, shape, r_min, r_max, dtype=jnp.float32):
    u = jax.random.uniform(key=key, shape=shape, dtype=dtype)
    return jnp.log(-0.5 * jnp.log(u * (r_max**2 - r_min**2) + r_min**2))


def theta_init(key, shape, max_phase, dtype=jnp.float32):
    u = jax.random.uniform(key, shape=shape, dtype=dtype)
    return jnp.log(max_phase * u)


def gamma_log_init(key, lamb):
    nu, theta = lamb
    diag_lambda = jnp.exp(-jnp.exp(nu) + 1j * jnp.exp(theta))
    return jnp.log(jnp.sqrt(1 - jnp.abs(diag_lambda) ** 2))


def matrix_init(key, shape, normalization, dtype=jnp.float32):
    return nn.initializers.glorot_uniform()(key, shape, dtype) / normalization


class ComplexDense(nn.Module):
    output_dim: int
    dtype: Any = jnp.float32
    use_bias: bool = True
    normalization: int = 1

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        W_real = nn.Dense(
            self.output_dim,
            dtype=self.dtype,
            use_bias=self.use_bias,
            kernel_init=lambda k, s, d: matrix_init(
                k, s, normalization=self.normalization, dtype=d
            ),
            name="real",
        )
        W_imag = nn.Dense(
            self.output_dim,
            dtype=self.dtype,
            use_bias=self.use_bias,
            kernel_init=lambda k, s, d: matrix_init(
                k, s, normalization=self.normalization, dtype=d
            ),
            name="imag",
        )
        return W_real(x) + 1j * W_imag(x)


class LRUCell(nn.Module):
    input_dim: int
    hidden_dim: int
    output_dim: int
    r_min: float = 0.0
    r_max: float = 1.0
    max_phase: float = 6.28
    dtype: Any = jnp.float32  # For real-valued parameters and inputs/outputs
    norm_before_readout: bool = True
    freeze_recurrence: bool = False

    def setup(self):
        if self.dtype == jnp.float64:
            print("WARNING: float64 not supported for LRUCell")
        dtype = jnp.float32

        # LRU parameters
        self.theta_log = self.param(
            "theta_log", partial(theta_init, max_phase=self.max_phase), (self.hidden_dim,)
        )
        self.nu_log = self.param(
            "nu_log", partial(nu_init, r_min=self.r_min, r_max=self.r_max), (self.hidden_dim,)
        )
        self.gamma_log = self.param("gamma_log", gamma_log_init, (self.nu_log, self.theta_log))

        self.B = ComplexDense(self.hidden_dim, dtype=self.dtype, normalization=jnp.sqrt(2))
        self.C = ComplexDense(self.hidden_dim, dtype=self.dtype)

        if self.norm_before_readout:
            self.layer_norm = nn.LayerNorm(dtype=dtype)

        self.mlp_readout = nn.Sequential(
            [
                nn.Dense(self.hidden_dim * 4, use_bias=True, dtype=dtype),
                nn.Dense(self.output_dim, use_bias=True, dtype=dtype),
            ]
        )

    def recurrence(self, h: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        # h is complex[H], x is real[O]
        diag_lambda = jnp.exp(-jnp.exp(self.nu_log) + 1j * jnp.exp(self.theta_log))
        gamma = jnp.exp(self.gamma_log)

        if self.freeze_recurrence:
            diag_lambda = jax.lax.stop_gradient(diag_lambda)

        # Recurrence: h_t+1 = lambda * h_t + B * x_t
        new_h = diag_lambda * h + gamma * self.B(x)
        if self.dtype == jnp.float32:
            return new_h
        else:
            return new_h.astype(jnp.complex128)

    def recurrence_jacobian(self, h: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        # Compute the Jacobian of the recurrence
        diag_lambda = jnp.exp(-jnp.exp(self.nu_log) + 1j * jnp.exp(self.theta_log))
        return jnp.diag(diag_lambda)

    def readout(self, h: jnp.ndarray) -> jnp.ndarray:
        # h is complex[H], x is real[O]
        y = self.C(h).real

        if self.norm_before_readout:
            y = self.layer_norm(y)

        y = self.mlp_readout(y)
        return y

    def __call__(self, carry: jnp.ndarray, inputs: jnp.ndarray) -> jnp.ndarray:
        x, h = inputs, carry
        new_h = self.recurrence(h, x)
        out = self.readout(new_h)
        return new_h, {"output": out}

    @nn.nowrap
    def initialize_carry(self, rng, input_shape) -> jnp.ndarray:
        dtype = jnp.complex128 if self.dtype == jnp.float64 else jnp.complex64
        return jnp.zeros((self.hidden_dim,), dtype=dtype)

    @nn.nowrap
    def carry_dtype(self, dtype):
        if dtype == jnp.float64:
            return jnp.complex128
        return jnp.complex64

    @nn.nowrap
    def is_complex(self):
        return True


class GRUCell(nn.Module):
    hidden_dim: int
    output_dim: int
    T_min: float = None
    T_max: float = None
    dtype: Any = jnp.float32
    norm_before_readout: bool = True

    def setup(self):
        dense_h = partial(
            nn.Dense,
            features=self.hidden_dim,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.dtype,
        )
        dense_i = partial(
            nn.Dense,
            features=self.hidden_dim,
            use_bias=True,
            dtype=self.dtype,
            param_dtype=self.dtype,
        )
        self.dense_iz = dense_i(
            name="iz",
            bias_init=partial(chrono_init, T_min=self.T_min, T_max=self.T_max),
        )
        self.dense_hz = dense_h(name="hz")
        self.dense_ir = dense_i(name="ir")
        self.dense_hr = dense_h(name="hr")
        self.dense_in = dense_i(name="in")
        self.dense_hn = dense_h(name="hn", use_bias=True)
        self.output_dense = nn.Dense(self.output_dim, name="output", dtype=self.dtype)
        if self.norm_before_readout:
            self.layer_norm = nn.LayerNorm(dtype=self.dtype)

    def recurrence(self, h: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        r = nn.sigmoid(self.dense_ir(x) + self.dense_hr(h))
        z = nn.sigmoid(self.dense_iz(x) + self.dense_hz(h))
        n = nn.tanh(self.dense_in(x) + r * self.dense_hn(h))
        new_h = (1.0 - z) * n + z * h
        return new_h

    def recurrence_jacobian(self, h: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        # Compute the Jacobian of the recurrence
        return jax.jacfwd(self.recurrence, argnums=0)(h, x)

    def readout(self, h: jnp.ndarray) -> jnp.ndarray:
        if self.norm_before_readout:
            h = self.layer_norm(h)
        return self.output_dense(h)

    def __call__(self, carry: jnp.ndarray, inputs: jnp.ndarray) -> jnp.ndarray:
        x, h = inputs, carry
        new_h = self.recurrence(h, x)
        out = self.readout(new_h)
        return new_h, {"output": out}

    @nn.nowrap
    def initialize_carry(self, rng, input_shape) -> jnp.ndarray:
        return jnp.zeros((self.hidden_dim,), dtype=self.dtype)

    @nn.nowrap
    def carry_dtype(self, dtype):
        return dtype

    @nn.nowrap
    def is_complex(self):
        return False


def cumulative_mean_pooling(x: jnp.ndarray) -> jnp.ndarray:
    """
    Straight-through cumulative mean pooling.
    """
    mean = jnp.cumsum(x, axis=0) / jnp.arange(1, x.shape[0] + 1)[:, None]
    return jax.lax.stop_gradient(mean - x) + x
