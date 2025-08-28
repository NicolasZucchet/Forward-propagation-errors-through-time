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

        is_complex = jnp.iscomplexobj(x)
        if is_complex:
            x_real, x_imag = x.real, x.imag
        else:
            x_real = x
    
        if is_complex:
            y_real = W_real(x_real) - W_imag(x_imag)
            y_imag = W_real(x_imag) + W_imag(x_real)
            return y_real + 1j * y_imag
        else:
            return W_real(x_real) + 1j * W_imag(x_real)


class LRUCell(nn.Module):
    hidden_dim: int
    output_dim: int
    r_min: float = 0.0
    r_max: float = 1.0
    max_phase: float = 6.28
    dtype: Any = jnp.float32  # For real-valued parameters and inputs/outputs
    norm_before_readout: bool = True
    freeze_recurrence: bool = False

    def setup(self):
        dtype = self.dtype

        # LRU parameters
        self.theta_log = self.param(
            "theta_log",
            partial(theta_init, max_phase=self.max_phase, dtype=dtype),
            (self.hidden_dim,),
        )
        self.nu_log = self.param(
            "nu_log",
            partial(nu_init, r_min=self.r_min, r_max=self.r_max, dtype=dtype),
            (self.hidden_dim,),
        )
        self.gamma_log = self.param("gamma_log", gamma_log_init, (self.nu_log, self.theta_log))

        self.B = ComplexDense(self.hidden_dim, dtype=dtype, normalization=jnp.sqrt(2))
        self.C = ComplexDense(self.hidden_dim, dtype=dtype)

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
        return new_h

    def recurrence_jacobian(self, h: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        # Compute the Jacobian of the recurrence
        diag_lambda = jnp.exp(-jnp.exp(self.nu_log) + 1j * jnp.exp(self.theta_log))
        return diag_lambda

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
    freeze_recurrence: bool = False

    def setup(self):
        if self.freeze_recurrence:
            raise NotImplementedError("Freezing recurrence is not implemented for GRUCell.")
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


class EUNNCell(nn.Module):
    """
    EUNN (tunable-space) using Algorithm 1 of the paper:

    For each layer:
      - v1 ← permute(v1, ind1)
      - v2 ← permute(v2, ind1)
      - y = v1 ⊙ x + v2 ⊙ permute(x, ind2)

    where ind1 and ind2 encode the Fa/Fb permutation patterns.
    """

    input_dim: int
    hidden_dim: int
    output_dim: int
    n_layers: int = 4
    dtype: Any = jnp.float32
    norm_before_readout: bool = True
    freeze_recurrence: bool = False
    nonlinearity: str = "none"  # "none", "tanh", or "modRelu"

    def setup(self):
        if self.dtype == jnp.float64:
            dtype = jnp.float64
        else:
            dtype = jnp.float32

        half = self.hidden_dim // 2

        def angle_init(key, shape, dtype=dtype):
            return jax.random.uniform(key, shape, dtype=dtype, minval=0.0, maxval=2 * jnp.pi)

        self.theta = self.param("theta", angle_init, (self.n_layers, half))
        self.phi = self.param("phi", angle_init, (self.n_layers, half))
        self.diag_phase = self.param("diag_phase", angle_init, (self.hidden_dim,))

        self.B = ComplexDense(self.hidden_dim, dtype=dtype, normalization=jnp.sqrt(2))
        self.C = ComplexDense(self.hidden_dim, dtype=dtype)

        if self.norm_before_readout:
            self.layer_norm = nn.LayerNorm(dtype=dtype)

        self.mlp_readout = nn.Sequential(
            [
                nn.Dense(self.hidden_dim * 4, use_bias=True, dtype=dtype),
                nn.Dense(self.output_dim, use_bias=True, dtype=dtype),
            ]
        )

        # Create modReLU bias once; unused if nonlinearity != modReLU
        self.modrelu_bias = self.param(
            "modrelu_bias",
            lambda k, s: jnp.zeros(s, dtype=dtype),
            (self.hidden_dim,),
        )


    def _apply_layers_vec(self, v: jnp.ndarray) -> jnp.ndarray:
        H = self.hidden_dim
        n_pairs = H // 2
        cdtype = jnp.complex128 if self.dtype == jnp.float64 else jnp.complex64

        # Apply diagonal phase D
        d = jnp.exp(1j * self.diag_phase)
        if self.freeze_recurrence:
            d = jax.lax.stop_gradient(d)
        v = d * v

        # Precompute both permutation patterns as arrays indexed by layer type
        # Shape: [2, H] where index 0 = Fa, index 1 = Fb
        
        # ind1: permutation for reordering v1, v2 coefficients
        # For tunable implementation, this depends on the layer pattern
        ind1_Fa = jnp.arange(H)  # Identity for Fa layers 
        ind1_Fb = jnp.arange(H)  
        ind1_Fb = jnp.roll(ind1_Fb, -1) # the first and the last are not used
        ind1_lookup = jnp.stack([ind1_Fa, ind1_Fb])
        
        # ind2: permutation for input vector (the actual pair swaps)
        # Fa: pairs (0,1), (2,3), (4,5), ... -> [1,0,3,2,5,4,...]
        ind2_Fa = jnp.arange(H)
        for i in range(0, H-1, 2):
            ind2_Fa = ind2_Fa.at[i].set(i+1)
            ind2_Fa = ind2_Fa.at[i+1].set(i)
        
        # Fb: pairs (1,2), (3,4), (5,6), ... -> [0,2,1,4,3,6,5,..., H-1] (the first at the last are not used)
        ind2_Fb = jnp.arange(H)
        for i in range(1, H-1, 2):
            ind2_Fb = ind2_Fb.at[i].set(i+1)
            if i+1 < H:
                ind2_Fb = ind2_Fb.at[i+1].set(i)
        # Stack into lookup tables
        ind2_lookup = jnp.stack([ind2_Fa, ind2_Fb])
        def layer_apply(vec, params):
            theta_l, phi_l, layer_idx = params
            if self.freeze_recurrence:
                theta_l = jax.lax.stop_gradient(theta_l)
                phi_l = jax.lax.stop_gradient(phi_l)

            c = jnp.cos(theta_l)        # [n_pairs]
            s = jnp.sin(theta_l)        # [n_pairs]
            e = jnp.exp(1j * phi_l)     # [n_pairs]

            # Build coefficient vectors with alternating pattern within pairs
            # v1 = (e^(iφ₁)cos θ₁, cos θ₁, e^(iφ₂)cos θ₂, cos θ₂, ...)
            # v2 = (-e^(iφ₁)sin θ₁, sin θ₁, -e^(iφ₂)sin θ₂, sin θ₂, ...)
            
            # Create empty arrays
            v1 = jnp.zeros(H, dtype=cdtype)
            v2 = jnp.zeros(H, dtype=cdtype)
            
            # Even indices (0, 2, 4, ...): with phase
            # Odd indices (1, 3, 5, ...): without phase
            even_idx = jnp.arange(0, H, 2)  # [0, 2, 4, ...]
            odd_idx = jnp.arange(1, H, 2)   # [1, 3, 5, ...]
            
            # Fill v1: even = e^(iφᵢ)cos θᵢ, odd = cos θᵢ
            v1 = v1.at[even_idx].set((c * e).astype(cdtype))    # e^(iφᵢ)cos θᵢ on even
            v1 = v1.at[odd_idx].set(c.astype(cdtype))           # cos θᵢ on odd
            
            # Fill v2: even = -e^(iφᵢ)sin θᵢ, odd = sin θᵢ  
            v2 = v2.at[even_idx].set((-s * e).astype(cdtype))   # -e^(iφᵢ)sin θᵢ on even
            v2 = v2.at[odd_idx].set(s.astype(cdtype))           # sin θᵢ on odd

            # Branch-free permutation selection using array indexing
            layer_type = layer_idx & 1  # 0 for Fa, 1 for Fb
            ind1 = ind1_lookup[layer_type]
            ind2 = ind2_lookup[layer_type]

            # Algorithm 1: Apply permutations to coefficient vectors
            # v1 ← permute(v1, ind1)
            # v2 ← permute(v2, ind1)
            v1 = v1[ind1]
            v2 = v2[ind1]

            # Algorithm 1: Apply transformation
            # y ← v1 * x + v2 * permute(x, ind2)
            # Support both shapes: [H] and [..., H] by operating on the last axis
            vec_perm = jnp.take(vec, ind2, axis=-1)
            y = v1 * vec + v2 * vec_perm 

            # In the case of Fb (branchless update; avoid jax.lax.select)
            mask = jnp.asarray((layer_type & 1) == 1, dtype=y.dtype)
            y = y.at[..., 0].set(y[..., 0] + mask * (vec[..., 0] - y[..., 0]))
            y = y.at[..., -1].set(y[..., -1] + mask * (vec[..., -1] - y[..., -1]))
            return y, None

        layers = (self.theta, self.phi, jnp.arange(self.n_layers))
        out, _ = jax.lax.scan(lambda vec, p: layer_apply(vec, (p[0], p[1], p[2])), v, layers)
        return out

    def _apply_nonlinearity(self, z: jnp.ndarray) -> jnp.ndarray:
        nl = (self.nonlinearity or "none").lower()
        if nl == "none":
            return z
        if nl == "tanh":
            return jnp.tanh(z)
        if nl == "modrelu":
            # b: learnable real bias per hidden unit
            b = self.modrelu_bias
            r = jnp.abs(z)
            scale = nn.relu(r + b) / (r + 1e-6)
            return z * scale
        raise ValueError(f"Unsupported EUNN nonlinearity: {self.nonlinearity}")
    def _unitary_matrix(self) -> jnp.ndarray:
        # Build U by applying layers to basis vectors (only when needed)
        H = self.hidden_dim
        cdtype = jnp.complex128 if self.dtype == jnp.float64 else jnp.complex64
        I = jnp.eye(H, dtype=cdtype)
        def col_apply(col):
            return self._apply_layers_vec(col)
        U = jax.vmap(col_apply, in_axes=1, out_axes=1)(I)
        return U

    def recurrence(self, h: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        v = self._apply_layers_vec(h)
        preact = v + self.B(x)
        new_h = self._apply_nonlinearity(preact)
        return new_h.astype(jnp.complex128 if self.dtype == jnp.float64 else jnp.complex64)

    def recurrence_jacobian(self, h: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        # J = D_nl(preact) @ U, where preact = U h + B x
        U = self._unitary_matrix()
        if self.freeze_recurrence:
            U = jax.lax.stop_gradient(U)
        preact = self._apply_layers_vec(h) + self.B(x)
        nl = (self.nonlinearity or "none").lower()
        if nl == "none":
            return U
        if nl == "tanh":
            deriv = 1.0 - jnp.tanh(preact) ** 2
            return deriv[:, None] * U
        if nl == "modrelu":
            b = self.modrelu_bias
            r = jnp.abs(preact)
            alpha = nn.relu(r + b) / (r + 1e-6)
            return alpha[:, None] * U
        raise ValueError(f"Unsupported EUNN nonlinearity: {self.nonlinearity}")

    def readout(self, h: jnp.ndarray) -> jnp.ndarray:
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
        return jnp.zeros((self.hidden_dim,), dtype=jnp.complex128 if self.dtype == jnp.float64 else jnp.complex64)

    @nn.nowrap
    def carry_dtype(self, dtype):
        return jnp.complex128 if dtype == jnp.float64 else jnp.complex64

    @nn.nowrap
    def is_complex(self):
        return True

