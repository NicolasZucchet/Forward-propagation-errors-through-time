"""
Forward backpropagation-through-time playground.
"""

import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn
from typing import Tuple, Callable, Any
from functools import partial
from omegaconf import DictConfig


def chrono_init(key, shape, dtype=jnp.float32, T_min=1.0, T_max=10.0):
    if T_min is None or T_max is None:
        return jnp.zeros(shape, dtype=dtype)
    return jnp.log(jax.random.uniform(key, shape, dtype=dtype, minval=T_min, maxval=T_max))


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


class StandardRNN(nn.Module):
    hidden_dim: int
    output_dim: int
    cell_type: nn.Module = GRUCell
    pooling: str = "none"  # "none" or "cumulative_mean"
    dtype: Any = jnp.float32
    unroll: int = 1

    @nn.compact
    def __call__(self, batch, init_carry=None):
        rnn = nn.scan(
            self.cell_type,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=0,
            out_axes=0,
            unroll=self.unroll,
        )(hidden_dim=self.hidden_dim, output_dim=self.output_dim, dtype=self.dtype)

        if init_carry is None:
            init_carry = rnn.initialize_carry(random.PRNGKey(0), batch["input"].shape)

        _, out = rnn(init_carry, batch["input"])

        if self.pooling == "cumulative_mean":
            out["output"] = cumulative_mean_pooling(out["output"])
        elif self.pooling != "none":
            raise ValueError(f"Unknown pooling type: {self.pooling}")

        return out  # {'output': [T, O]}


class ForwardBPTTCell(nn.Module):
    hidden_dim: int
    output_dim: int
    loss_fn: Callable
    cell_type: nn.Module = GRUCell
    dtype: Any = jnp.float32
    approx_inverse: bool = False  # Use approximate inverse for delta update
    norm_before_readout: bool = True
    pooling: str = "none"

    @nn.compact
    def __call__(self, carry: Any, inputs: jnp.ndarray):
        x, y, m = inputs  # x_t+1, y_t+1, m_t+1 (mask)
        h, delta, inst_delta, prod_jac, prev_mean, t = (
            carry  # h_t, delta_t, inst_delta_t, prod_jac_t, mean_t, t
        )

        cell = self.cell_type(
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            dtype=self.dtype,
            norm_before_readout=self.norm_before_readout,
        )

        # New hidden state (x: [X], h: [H])
        new_h = cell.recurrence(h, x)  # h_t+1: [H]
        new_t = t + 1.0
        if self.pooling == "cumulative_mean":
            new_mean = (t * prev_mean + new_h) / new_t
        elif self.pooling == "none":
            new_mean = new_h  # Not used
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling}")
        out = cell.readout(new_mean)  # pred_t+1: [O]

        # New jacobian product: prod_t+1 = J_t^T @ prod_t
        jacobian = jax.jacfwd(cell.recurrence, argnums=0)(
            h, x
        )  # [H, H], jacobian of the hidden state update
        jacobian = jacobian.transpose()  # [H, H]
        new_prod_jac = prod_jac @ jacobian  # [H, H]

        # New delta: delta_t+1 = (J_t^T)^{-1} (delta_t - inst_delta_t)
        # NOTE: it requires the instantaneous delta computed in the previous iteration!
        # Depending on the approx_inverse flag, either compute the exact inverse or an approximation
        if not self.approx_inverse:
            new_delta = jnp.linalg.inv(jacobian) @ (delta - inst_delta)  # [H], reverse BPTT update
        else:
            # Approximate the inverse Jacobian assuming it is close to identity
            # (J_t^T)^{-1} = (Id + (J_t^T - Id))^{-1} ~ 2Id - J_t^T
            new_delta = 2 * (delta - inst_delta) - jacobian @ (delta - inst_delta)  # [H]

            # TODO: more fancy approximations could be used like Newton-Schulz
            # In principle it would only require jvps, although we need the full jacobian to compute
            # delta_0

        # We compute the instantaneous delta for the next iteration as information is available now
        # NOTE: giving the current mean corresponds to having a straight-though cumulative mean
        # pooling
        new_inst_delta = jax.grad(lambda _h: self.loss_fn(cell.readout(_h), y, m))(new_mean)  # [H]

        new_carry = new_h, new_delta, new_inst_delta, new_prod_jac, new_mean, new_t
        out = {
            "output": out,  # pred_t+1
            "prev_h": h,  # h_t
            "delta": new_delta,  # delta_t+1
            "inst_delta": new_inst_delta,  # inst_delta_t+1
            "delta_output": jax.grad(lambda o: self.loss_fn(o, y, m))(out),
            "h_norm": jnp.linalg.norm(new_h),
            "delta_norm": jnp.linalg.norm(new_delta),
            "prod_jac_norm": jnp.linalg.norm(new_prod_jac),
            "jac": jacobian,
        }
        return new_carry, out

    def initialize_carry(self, rng, input_shape) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        h = jnp.zeros((self.hidden_dim,), dtype=self.dtype)
        delta = jnp.zeros((self.hidden_dim,), dtype=self.dtype)
        inst_delta = jnp.zeros((self.hidden_dim,), dtype=self.dtype)
        prod_jac = jnp.eye(self.hidden_dim, dtype=self.dtype)
        prev_mean = jnp.zeros((self.hidden_dim,), dtype=self.dtype)
        t = 0.0
        return h, delta, inst_delta, prod_jac, prev_mean, t


class ForwardBPTTRNN(nn.Module):
    hidden_dim: int
    output_dim: int
    cell: nn.Module = ForwardBPTTCell
    dtype: Any = jnp.float32
    two_passes: bool = True  # start with non zero, correct delta_0
    norm_before_readout: bool = True
    pooling: str = "none"  # "none" or "cumulative_mean"
    unroll: int = 1

    @nn.compact
    def __call__(self, batch):
        if self.dtype == jnp.float32:
            print(
                "WARNING: Using float32 precision may lead to numerical instability for forward BPTT!"
            )

        inputs, targets, masks = batch["input"], batch["target"], batch["mask"]

        # Extended forward pass to include forward BPTT equations + how to use those results to
        # compute gradients
        rnn = nn.scan(
            self.cell,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=0,
            out_axes=0,
            length=inputs.shape[0],
            unroll=self.unroll,
        )(
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            dtype=self.dtype,
            norm_before_readout=self.norm_before_readout,
        )

        cell = self.cell(hidden_dim=self.hidden_dim, output_dim=self.output_dim, dtype=self.dtype)
        cell = cell.cell_type(
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            dtype=self.dtype,
            norm_before_readout=self.norm_before_readout,
        )

        def f(module, x, y, m):
            # If the module is just used for forward pass, just do one pass and return the output.
            init_carry = rnn.initialize_carry(None, None)
            _, out = module(init_carry, (x, y, m))
            return out

        def fwd(module, x, y, m):
            init_carry = rnn.initialize_carry(None, None)
            if self.two_passes:
                # Step 1: run extended forward pass starting from the initial carry.
                final_carry, _ = module(init_carry, (x, y, m))

                # Step 2: compute the new initial carry (in particular delta) and perform the second pass
                _, final_delta, last_inst_delta, final_prod_jac, _, _ = final_carry
                # We use that:
                #   1. delta_t - delta_t^BP = -prod_t' J_t'^{-T} delta_0^BP
                #   2. delta_T^BP = inst_delta_T
                # So that
                # delta_0^BP = - (prod_t' J_t'^T) (delta_T - inst_delta_T)
                delta_0 = -final_prod_jac @ (final_delta - last_inst_delta)

            else:
                # Just start directly at 0. NOTE: this will not yield the true gradient
                delta_0 = jnp.zeros((self.hidden_dim,), dtype=self.dtype)

            new_carry = tuple(init_carry[i] if i != 1 else delta_0 for i in range(len(init_carry)))
            final_carry, out = module(new_carry, (x, y, m))

            # To check wether that after the second pass, the final delta matches with the one of
            # BPTT, that is the final instantaneous delta.
            residual_error = final_carry[1] - final_carry[2]  # Useful for debugging
            residual_error_delta = jnp.linalg.norm(residual_error)
            norm_prod_jac = jnp.linalg.norm(final_carry[3])

            # Step 3: get the JVP function to do error -> delta mapping. Use the vanilla cell for that.
            def _fn(p):
                _h, _o = cell.apply({"params": p}, out["prev_h"], x)
                return _h, _o["output"]

            _, vjp = jax.vjp(_fn, module.variables["params"]["GRUCell_0"])

            # Gather some additional logging information
            fn_out = {
                "output": out["output"],  # predictions
                "norm_prod_jac": norm_prod_jac,
                "residual_error_delta": residual_error_delta,
                "norm_delta_0": jnp.linalg.norm(delta_0),
            }  # We don't add all outputs here to avoid keeping too much information in memory.

            return fn_out, (vjp, out)

        def bwd(from_forward, _):  # NOTE: we ignore errors received from downstream graph
            vjp, info = from_forward
            delta_h = info["delta"]  # [T, H]
            delta_out = info["delta_output"]  # [T, O]

            # Step 4: compute the parameter grad for all output parameters.
            grad_params_output = vjp((jnp.zeros_like(delta_h), delta_out))[0]  # PARAMS

            # Step 5: compute the parameter grad for all the other parameters
            grad_params_hidden = vjp((delta_h, jnp.zeros_like(delta_out)))[0]
            # NOTE we do that because we overwrite the internal propagation of errors
            # TODO: would be possible to avoid it if stop grad before readout

            # Step 6: merge
            grad_params = jax.tree.map(
                lambda a, b: (a - b) * (jnp.linalg.norm(b) < 1e-8) + b,
                grad_params_output,
                grad_params_hidden,
            )

            return (
                {"params": {"GRUCell_0": grad_params}},
                jnp.zeros_like(inputs),
                jnp.zeros_like(targets),
                jnp.zeros_like(masks),
            )

        custom_f = nn.custom_vjp(fn=f, forward_fn=fwd, backward_fn=bwd)
        return custom_f(rnn, inputs, targets, masks)


def create_model(
    cfg: DictConfig,
    output_dim: int,
    seq_len: int,
    loss_fn: Callable,
    dtype: Any,
    batch: dict,
    key: jax.random.PRNGKey,
):
    """
    Helper function to create a model and convert parameters if needed.
    """
    assert cfg.model.training_mode in ["normal", "forward", "forward_forward"]
    cell_type = partial(
        GRUCell,
        T_min=seq_len * cfg.model.T_min_frac if cfg.model.T_min_frac is not None else None,
        T_max=seq_len * cfg.model.T_max_frac if cfg.model.T_max_frac is not None else None,
        norm_before_readout=cfg.model.norm_before_readout,
        dtype=dtype,
    )  # Long time scales to give forward BPTT a chance
    BatchedRNN = nn.vmap(
        partial(
            StandardRNN,
            cell_type=cell_type,
            pooling=cfg.model.pooling,
            dtype=dtype,
            unroll=cfg.model.unroll,
        ),
        in_axes=0,
        out_axes=0,
        variable_axes={"params": None},
        split_rngs={"params": False},
    )
    batched_model = BatchedRNN(hidden_dim=cfg.model.hidden_dim, output_dim=output_dim, dtype=dtype)
    params = batched_model.init(key, batch)["params"]

    if cfg.model.training_mode in ["forward", "forward_forward"]:
        # Overwrite the model to use the correct one, and convert parameters
        model = partial(
            ForwardBPTTRNN,
            cell=partial(
                ForwardBPTTCell,
                cell_type=cell_type,
                loss_fn=loss_fn,
                dtype=dtype,
                approx_inverse=cfg.model.approx_inverse,
                norm_before_readout=cfg.model.norm_before_readout,
                pooling=cfg.model.pooling,
            ),
            dtype=dtype,
            two_passes=cfg.model.training_mode == "forward_forward",
            pooling=cfg.model.pooling,
        )
        BatchedRNN = nn.vmap(
            model,
            in_axes=0,
            out_axes=0,
            variable_axes={"params": None},
            split_rngs={"params": False},
        )
        batched_model = BatchedRNN(
            hidden_dim=cfg.model.hidden_dim, output_dim=output_dim, dtype=dtype
        )
        params = conversion_params_normal_to_forwardbptt(params)

    return params, batched_model


def conversion_params_normal_to_forwardbptt(params: dict) -> dict:
    """
    Convert parameters from StandardRNN to ForwardBPTTRNN format.
    """
    return {"ScanForwardBPTTCell_0": {"GRUCell_0": params["ScanGRUCell_0"]}}


def cumulative_mean_pooling(x: jnp.ndarray) -> jnp.ndarray:
    """
    Straight-through cumulative mean pooling.
    """
    mean = jnp.cumsum(x, axis=0) / jnp.arange(1, x.shape[0] + 1)[:, None]
    return jax.lax.stop_gradient(mean - x) + x
