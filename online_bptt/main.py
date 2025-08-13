"""
Forward backpropagation-through-time playground.
"""

import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn
from typing import Tuple, Callable, Any
from functools import partial

# NOTE: important to set JAX to use 64-bit precision for numerical stability when dealing with
# inverse of matrices that can explode
jax.config.update("jax_enable_x64", True)


class GRUCell(nn.Module):
    hidden_dim: int
    output_dim: int

    def setup(self):
        dense_h = partial(
            nn.Dense,
            features=self.hidden_dim,
            use_bias=False,
        )
        dense_i = partial(
            nn.Dense,
            features=self.hidden_dim,
            use_bias=True,
        )
        self.dense_iz = dense_i(name="iz")
        self.dense_hz = dense_h(name="hz")
        self.dense_ir = dense_i(name="ir")
        self.dense_hr = dense_h(name="hr")
        self.dense_in = dense_i(name="in")
        self.dense_hn = dense_h(name="hn", use_bias=True)
        self.output_dense = nn.Dense(self.output_dim, name="output")

    def recurrence(self, h: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        r = nn.sigmoid(self.dense_ir(x) + self.dense_hr(h))
        z = nn.sigmoid(self.dense_iz(x) + self.dense_hz(h))
        n = nn.tanh(self.dense_in(x) + r * self.dense_hn(h))
        new_h = (1.0 - z) * n + z * h
        return new_h

    def readout(self, h: jnp.ndarray) -> jnp.ndarray:
        return self.output_dense(h)

    def __call__(self, carry: jnp.ndarray, inputs: jnp.ndarray) -> jnp.ndarray:
        x, h = inputs, carry
        new_h = self.recurrence(h, x)
        out = self.readout(new_h)
        return new_h, out

    @nn.nowrap
    def initialize_carry(self, rng, input_shape) -> jnp.ndarray:
        return jnp.zeros((self.hidden_dim,))


class StandardRNN(nn.Module):
    hidden_dim: int
    output_dim: int
    cell_type: nn.Module = GRUCell

    @nn.compact
    def __call__(self, inputs, init_carry=None):
        rnn = nn.scan(
            self.cell_type,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=0,
            out_axes=0,
        )(hidden_dim=self.hidden_dim, output_dim=self.output_dim)

        if init_carry is None:
            init_carry = rnn.initialize_carry(random.PRNGKey(0), inputs.shape)

        _, outputs = rnn(init_carry, inputs)
        return self.perturb("outputs", outputs)


class ForwardBPTTCell(nn.Module):
    hidden_dim: int
    output_dim: int
    loss_fn: Callable
    cell_type: nn.Module = GRUCell

    @nn.compact
    def __call__(self, carry: Any, inputs: jnp.ndarray):
        x, y = inputs  # x_t+1, y_t+1
        h, delta, inst_delta, prod_jac = carry  # h_t, delta_t, inst_delta_t, prod_jac_t

        cell = self.cell_type(hidden_dim=self.hidden_dim, output_dim=self.output_dim)

        # New hidden state (x: [X], h: [H])
        new_h = cell.recurrence(h, x)  # h_t+1: [H]
        out = cell.readout(new_h)  # pred_t+1: [O]

        # New jacobian product: prod_t+1 = J_t^T @ prod_t
        jacobian = jax.jacfwd(cell.recurrence, argnums=0)(
            h, x
        )  # [H, H], jacobian of the hidden state update
        jacobian = jacobian.transpose()  # [H, H]
        new_prod_jac = prod_jac @ jacobian  # [H, H]

        # New delta: delta_t+1 = (J_t^T)^{-1} (delta_t - inst_delta_t)
        # NOTE: it requires the instantaneous delta computed in the previous iteration!
        new_delta = jnp.linalg.inv(jacobian) @ (delta - inst_delta)  # [H], reverse BPTT update

        # We compute the instantaneous delta for the next iteration as information is available now
        new_inst_delta = jax.grad(lambda _h: self.loss_fn(cell.readout(_h), y))(new_h)  # [H]

        new_carry = new_h, new_delta, new_inst_delta, new_prod_jac
        out = {
            "output": out,  # pred_t+1
            "prev_h": h,  # h_t
            "delta": new_delta,  # delta_t+1
            "inst_delta": new_inst_delta,  # inst_delta_t+1
            "delta_output": jax.grad(lambda o: self.loss_fn(o, y))(out),
            "h_norm": jnp.linalg.norm(new_h),
            "delta_norm": jnp.linalg.norm(new_delta),
            "prod_jac_norm": jnp.linalg.norm(new_prod_jac),
            "jac": jacobian,
        }
        return new_carry, out

    def initialize_carry(self, rng, input_shape) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        h = jnp.zeros((self.hidden_dim,))
        delta = jnp.zeros((self.hidden_dim,))
        inst_delta = jnp.zeros((self.hidden_dim,))
        prod_jac = jnp.eye(self.hidden_dim)
        return h, delta, inst_delta, prod_jac


class ForwardBPTTRNN(nn.Module):
    hidden_dim: int
    output_dim: int
    cell: nn.Module = ForwardBPTTCell

    @nn.compact
    def __call__(self, inputs, targets):
        # Extended forward pass to include forward BPTT equations + how to use those results to
        # compute gradients
        rnn = nn.scan(
            self.cell,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=0,
            out_axes=0,
            length=inputs.shape[0],
        )(hidden_dim=self.hidden_dim, output_dim=self.output_dim)

        cell = self.cell(hidden_dim=self.hidden_dim, output_dim=self.output_dim)
        cell = cell.cell_type(hidden_dim=self.hidden_dim, output_dim=self.output_dim)

        def f(module, x, y):
            # If the module is just used for forward pass, just do one pass and return the output.
            init_carry = rnn.initialize_carry(None, None)
            _, out = module(init_carry, (x, y))
            return out["output"]

        def fwd(module, x, y):
            # Step 1: run extended forward pass starting from the initial carry.
            init_carry = rnn.initialize_carry(None, None)
            final_carry, _ = module(init_carry, (x, y))

            # Step 2: compute the new initial carry (in particular delta) and perform the second pass
            _, final_delta, last_inst_delta, final_prod_jac = final_carry
            # We use that:
            #   1. delta_t - delta_t^BP = -prod_t' J_t'^{-T} delta_0^BP
            #   2. delta_T^BP = inst_delta_T
            # So that
            # delta_0^BP = - (prod_t' J_t'^T) (delta_T - inst_delta_T)
            new_delta = -final_prod_jac @ (final_delta - last_inst_delta)

            new_carry = (init_carry[0], new_delta, init_carry[2], init_carry[3])
            final_carry, out = module(new_carry, (x, y))

            # Check that after the second pass, the final delta matches with the one of BPTT, that
            # is the final instantaneous delta.
            assert (
                jnp.linalg.norm(final_carry[1] - final_carry[2]) < 1e-4
            ), "Final delta does not match, discrepancy: {}".format(
                jnp.linalg.norm(final_carry[1] - final_carry[2])
            )

            # Step 3: get the JVP function to do error -> delta mapping. Use the vanilla cell for that.
            _, vjp = jax.vjp(
                lambda p: cell.apply({"params": p}, out["prev_h"], x),
                module.variables["params"]["GRUCell_0"],
            )

            return out["output"], (vjp, out)

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
            )

        custom_f = nn.custom_vjp(fn=f, forward_fn=fwd, backward_fn=bwd)
        return custom_f(rnn, inputs, targets)


