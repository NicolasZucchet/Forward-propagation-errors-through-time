import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, Callable, Any
from online_bptt.model.cells import GRUCell, cumulative_mean_pooling


class StandardRNN(nn.Module):
    hidden_dim: int
    output_dim: int
    cell_type: nn.Module = GRUCell
    pooling: str = "none"  # "none" or "cumulative_mean" or "mean"
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
            init_carry = rnn.initialize_carry(jax.random.PRNGKey(0), batch["input"].shape)

        _, out = rnn(init_carry, batch["input"])

        if self.pooling == "cumulative_mean":
            out["output"] = cumulative_mean_pooling(out["output"])
        elif self.pooling == "mean":
            out["output"] = (
                jnp.cumsum(out["output"], axis=0)
                / jnp.arange(1, out["output"].shape[0] + 1)[:, None]
            )  # cumulative mean without stop gradient. NOTE: non-causal
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

    def setup(self):
        self.cell = self.cell_type(
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            dtype=self.dtype,
            norm_before_readout=self.norm_before_readout,
        )

    def __call__(self, carry: Any, inputs: jnp.ndarray):
        x, y, m = inputs  # x_t+1, y_t+1, m_t+1 (mask)
        h, delta, inst_delta, prod_jac, prev_mean, t = (
            carry  # h_t, delta_t, inst_delta_t, prod_jac_t, mean_t, t
        )

        # New hidden state (x: [X], h: [H])
        new_h = self.cell.recurrence(h, x)  # h_t+1: [H]
        new_t = t + 1.0
        if self.pooling == "cumulative_mean":
            new_mean = (t * prev_mean + new_h) / new_t
        elif self.pooling == "none":
            new_mean = new_h  # Not used
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling}")
        out = self.cell.readout(new_mean)  # pred_t+1: [O]

        # New jacobian product: prod_t+1 = J_t^T @ prod_t
        jacobian = self.cell.recurrence_jacobian(
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
        new_inst_delta = jax.grad(lambda _h: self.loss_fn(self.cell.readout(_h), y, m))(
            new_mean
        )  # [H]

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

    def initialize_carry(
        self, rng, input_shape, dtype=None
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        if dtype is None:
            dtype = self.cell.carry_dtype(self.dtype)
        h = jnp.zeros((self.hidden_dim,), dtype=dtype)
        delta = jnp.zeros((self.hidden_dim,), dtype=dtype)
        inst_delta = jnp.zeros((self.hidden_dim,), dtype=dtype)
        prod_jac = jnp.eye(self.hidden_dim, dtype=dtype)
        prev_mean = jnp.zeros((self.hidden_dim,), dtype=dtype)
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

            new_carry = tuple(
                init_carry[i] if i != 1 else delta_0.astype(init_carry[i].dtype)
                for i in range(len(init_carry))
            )
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

            _, vjp = jax.vjp(_fn, module.variables["params"]["cell"])

            # Gather some additional logging information
            fn_out = {
                "output": out["output"],  # predictions
                "norm_prod_jac": norm_prod_jac,
                "residual_error_delta": residual_error_delta,
                "norm_delta_0": jnp.linalg.norm(delta_0),
                "norm_final_delta_first_pass": jnp.linalg.norm(
                    final_delta if self.two_passes else final_carry[1]
                ),
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
                {"params": {"cell": grad_params}},
                jnp.zeros_like(inputs),
                jnp.zeros_like(targets),
                jnp.zeros_like(masks),
            )

        custom_f = nn.custom_vjp(fn=f, forward_fn=fwd, backward_fn=bwd)
        return custom_f(rnn, inputs, targets, masks)
