import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, Callable, Any
from online_bptt.model.cells import GRUCell, LRUCell, cumulative_mean_pooling
from functools import partial
from utils import get_logger

logger = get_logger()


class StandardLayer(nn.Module):
    hidden_dim: int
    output_dim: int
    cell_type: nn.Module = GRUCell
    dtype: Any = jnp.float32
    unroll: int = 1
    stop_gradients: str = "none"

    def setup(self):
        self.layer = nn.scan(
            self.cell_type,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=0,
            out_axes=0,
            unroll=self.unroll,
        )(
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            dtype=self.dtype,
            stop_gradients=self.stop_gradients,
        )

    def __call__(self, x, init_carry=None):
        if init_carry is None:
            init_carry = self.initialize_carry(None, None)

        return self.layer(init_carry, x)[1]  # {'output': [T, O]}

    def initialize_carry(self, key, input_shape):
        return self.layer.initialize_carry(key, input_shape)


class ForwardBPTTCell(nn.Module):
    hidden_dim: int
    output_dim: int
    cell_type: nn.Module = GRUCell
    base_precision: Any = jnp.float32
    increased_precision: Any = jnp.float64
    approx_inverse: bool = False  # Use approximate inverse for delta update
    norm_before_readout: bool = True
    pooling: str = "none"

    def setup(self):
        self.cell = self.cell_type(
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            dtype=self.base_precision,
            norm_before_readout=self.norm_before_readout,
        )
        self.diagonal_jacobian = isinstance(self.cell, LRUCell)

    def __call__(self, carry: Any, inputs: jnp.ndarray):
        x, delta_out = inputs  # x_t+1, delta_out_t+1 (error at output)
        h, delta, inst_delta, prod_jac = carry  # h_t, delta_t, inst_delta_t, prod_jac_t

        # New hidden state (x: [X], h: [H])
        new_h = self.cell.recurrence(h, x)  # h_t+1: [H]
        out = self.cell.readout(new_h)  # pred_t+1: [O]

        # New jacobian product: prod_t+1 = prod_t @ J_t^T
        jacobian = self.cell.recurrence_jacobian(
            h, x
        )  # [H, H] or [H], jacobian of the hidden state update
        if self.diagonal_jacobian:
            # NOTE: no need to transpose here
            new_prod_jac = prod_jac * jacobian  # [H]
        else:
            jacobian = jacobian.transpose()  # [H, H]
            new_prod_jac = prod_jac @ jacobian  # [H, H]

        # New delta: delta_t+1 = (J_t^T)^{-1} (delta_t - inst_delta_t)
        # NOTE: it requires the instantaneous delta computed in the previous iteration!
        # Depending on the approx_inverse flag, either compute the exact inverse or an approximation
        if not self.approx_inverse:
            # new_delta: [H], reverse BPTT update
            if self.diagonal_jacobian:
                # leverage that the jacobian is diagonal
                new_delta = (delta - inst_delta) / jacobian
            else:
                inv_jacobian = jnp.linalg.inv(jacobian)
                new_delta = inv_jacobian @ (delta - inst_delta)
        else:
            # Approximate the inverse Jacobian assuming it is close to identity
            # (J_t^T)^{-1} = (Id + (J_t^T - Id))^{-1} ~ 2Id - J_t^T
            new_delta = 2 * (delta - inst_delta) - jacobian @ (delta - inst_delta)  # [H]

            # TODO: more fancy approximations could be used like Newton-Schulz
            # In principle it would only require jvps, although we need the full jacobian to compute
            # delta_0

        # We compute the instantaneous delta for the next iteration as information is available now
        (new_inst_delta,) = jax.vjp(self.cell.readout, new_h)[1](delta_out)  # [H]

        new_carry = new_h, new_delta, new_inst_delta, new_prod_jac
        out = {
            "output": out,  # pred_t+1
            "prev_h": h,  # h_t
            "delta": new_delta,  # delta_t+1
            "inst_delta": new_inst_delta,  # inst_delta_t+1
            "h_norm": jnp.linalg.norm(new_h),
            "delta_norm": jnp.linalg.norm(new_delta),
            "prod_jac_norm": jnp.linalg.norm(new_prod_jac),
            "jac": jacobian,
        }
        return new_carry, out

    def carry_dtypes(self, dtype=None):
        if dtype is None:
            # Taking into account whether the cell uses complex number or not
            carry_base_precision = self.cell.carry_dtype(self.base_precision)
            carry_increased_precision = self.cell.carry_dtype(self.increased_precision)
            return {
                "h": carry_base_precision,
                "delta": carry_increased_precision,  # we need higher precision here as this quantity can explode
                "inst_delta": carry_base_precision,
                "prod_jac": carry_base_precision,
            }
        else:
            return {k: dtype for k in ["h", "delta", "inst_delta", "prod_jac"]}

    def initialize_carry(
        self, rng, input_shape, dtype=None, diagonal_jacobian=None
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        dtypes = self.carry_dtypes(dtype)

        # Whether the Jacobian is diagonal
        diagonal_jacobian = (
            diagonal_jacobian if diagonal_jacobian is not None else self.diagonal_jacobian
        )

        h = jnp.zeros((self.hidden_dim,), dtype=dtypes["h"])
        delta = jnp.zeros((self.hidden_dim,), dtype=dtypes["delta"])
        inst_delta = jnp.zeros((self.hidden_dim,), dtype=dtypes["inst_delta"])
        if diagonal_jacobian:
            prod_jac = jnp.ones((self.hidden_dim,), dtype=dtypes["prod_jac"])
        else:
            prod_jac = jnp.eye(self.hidden_dim, dtype=dtypes["prod_jac"])
        return h, delta, inst_delta, prod_jac


class ForwardBPTTLayer(nn.Module):
    hidden_dim: int
    output_dim: int
    length: int
    cell: nn.Module = ForwardBPTTCell
    base_precision: Any = jnp.float32
    increased_precision: Any = jnp.float64
    two_passes: bool = True  # start with non zero, correct delta_0
    norm_before_readout: bool = True
    unroll: int = 1

    def setup(self):
        if self.increased_precision == jnp.float32:
            print(
                "WARNING: Using float32 precision may lead to numerical instability for forward BPTT!"
            )

        # Extended forward pass to include forward BPTT equations + how to use those results to
        # compute gradients
        self.layer = nn.scan(
            self.cell,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=0,
            out_axes=0,
            length=self.length,
            unroll=self.unroll,
        )(
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            base_precision=self.base_precision,
            increased_precision=self.increased_precision,
            norm_before_readout=self.norm_before_readout,
        )
        self.diagonal_jacobian = self.layer.diagonal_jacobian

        cell = self.cell(
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            base_precision=self.base_precision,
            increased_precision=self.increased_precision,
        )  # HACK: only instantiate the ForwardBPTTCell to get access to the inner cell
        self.inner_cell = cell.cell_type(
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            dtype=self.base_precision,
            norm_before_readout=self.norm_before_readout,
            stop_gradients="readout",  # NOTE: needed for bwd
        )

        # WARNING: it is important to setup the base precision in the setup (vs in a nn.compact
        # call) to avoid closure problems as it is needed in the custom backward pass and jax does
        # not support sending dtypes through the forward pass.
        self.base_carry_precision = self.inner_cell.carry_dtype(self.base_precision)

    def __call__(self, x):

        def f(module, x):
            # If the module is just used for forward pass, just do one pass and return the output.
            init_carry = module.initialize_carry(None, None)
            _delta_out = jnp.zeros(
                (x.shape[0], self.output_dim), dtype=self.base_precision
            )  # fake error
            _, out = module(init_carry, (x, _delta_out))
            return out

        def fwd(module, x):
            # Run forward pass to get the output and the hidden states
            out = f(module, x)

            # Use the vanilla cell to get the delta to gradients mapping
            def _fn(_p, _x):
                _h, _o = self.inner_cell.apply({"params": _p}, out["prev_h"], _x)
                return _h, _o["output"]

            _, vjp = jax.vjp(_fn, module.variables["params"]["cell"], x)

            # For closure reasons, we need to directly give the backward pass everything it needs.
            # This includes jvp, forward pass through the ForwardBPTTCell layer and inputs.
            to_bwd = (vjp, jax.tree_util.Partial(module.__call__), x)

            return out, to_bwd

        def bwd(from_forward, delta):
            vjp, fwd_pass, x = from_forward
            dtypes = self.layer.carry_dtypes()
            delta_out = delta["output"]

            # 1. First forward pass to determine delta_0
            init_carry = self.layer.initialize_carry(None, None)
            if self.two_passes:
                # Run extended forward pass starting from the initial carry.
                final_carry, out = fwd_pass(init_carry, (x, delta_out))

                # Use the first pass to compute the initial delta
                _, final_delta, last_inst_delta, final_prod_jac = final_carry
                # We use that:
                #   - delta_t - delta_t^BP = -prod_t' J_t'^{-T} delta_0^BP
                #   - delta_T^BP = inst_delta_T
                # So that
                # delta_0^BP = - (prod_t' J_t'^T) (delta_T - inst_delta_T)
                if self.diagonal_jacobian:
                    delta_0 = -final_prod_jac * (final_delta - last_inst_delta)
                else:
                    delta_0 = -final_prod_jac @ (final_delta - last_inst_delta)

                # Log extremum values of delta
                max_delta_1 = jnp.max(jnp.abs(out["delta"]))
                mean_delta_1 = jnp.mean(jnp.abs(out["delta"]))

            else:
                # Just start directly at delta=0.
                # NOTE: this will not yield the true gradient
                delta_0 = jnp.zeros((self.hidden_dim,), dtype=dtypes["delta"])
                mean_delta_1, max_delta_1 = None, None

            # 2. Second forward pass to compute the actual delta trajectory
            new_carry = tuple(init_carry[i] if i != 1 else delta_0 for i in range(len(init_carry)))
            final_carry, out = fwd_pass(new_carry, (x, delta_out))

            # Check that wether final delta matches with the one of BPTT, that is the final instantaneous delta.
            residual_error = final_carry[1] - final_carry[2]  # Useful for debugging, should be 0
            residual_error_delta = jnp.linalg.norm(residual_error)
            norm_prod_jac = jnp.linalg.norm(final_carry[3])

            # 3. Compute the actual gradients
            delta_h = out["delta"]  # [T, H]
            # NOTE: when using vjps, we use the base precision. The increased precision is only used
            # when computing the deltas
            delta_h = delta_h.astype(self.base_carry_precision)

            # NOTE: having stop_gradients=readout is important, otherwise the parameters within the
            # recurrence also receive the delta_out backpropagated through the recurrence
            grad_params, grad_inputs = vjp((delta_h, delta_out))  # PARAMS

            # Log some metrics about the backward pass
            logs = {
                f"log/{self.name}/norm_prod_jac": norm_prod_jac,
                f"log/{self.name}/residual_error_delta": residual_error_delta,
                f"log/{self.name}/norm_delta_0": jnp.linalg.norm(delta_0),
                f"log/{self.name}/norm_final_delta_first_pass": jnp.linalg.norm(
                    final_delta if self.two_passes else final_carry[1]
                ),
                f"log/{self.name}/max_delta_1": max_delta_1,
                f"log/{self.name}/mean_delta_1": mean_delta_1,
                f"log/{self.name}/min_delta_2": jnp.mean(jnp.abs(delta_h)),
                f"log/{self.name}/max_delta_2": jnp.max(jnp.abs(delta_h)),
                f"log/{self.name}/mean_delta_2": jnp.mean(jnp.abs(delta_h)),
            }
            jax.debug.callback(logger.log_callback, logs)

            return {"params": {"cell": grad_params}}, grad_inputs

        custom_f = nn.custom_vjp(fn=f, forward_fn=fwd, backward_fn=bwd)
        return custom_f(self.layer, x)


class RNN(nn.Module):
    hidden_dim: int
    output_dim: int
    training_mode: str
    cell_type: nn.Module = GRUCell
    n_layers: int = 1
    pooling: str = "none"  # "none" or "cumulative_mean" or "mean"
    dtype: Any = jnp.float32
    unroll: int = 1
    base_precision: Any = jnp.float32
    increased_precision: Any = jnp.float64
    two_passes: bool = True
    approx_inverse: bool = False
    norm_before_readout: bool = True
    forward_simulation_passes: int = 2  # None

    @nn.compact
    def __call__(self, x):
        if self.training_mode in ["normal", "spatial"]:
            layers = [StandardLayer] * self.n_layers
            kwargs = [
                {
                    "cell_type": self.cell_type,
                    "dtype": self.dtype,
                    "unroll": self.unroll,
                    "stop_gradients": "none" if self.training_mode == "normal" else "spatial",
                }
            ] * self.n_layers
        elif self.training_mode in ["forward", "forward_forward"]:
            # If forward_simulation_passes is set, we simulate running the algorithm for a given
            # number of forward passes
            # If 1, all layers are in spatial mode
            # If 2, all layers are in spatial mode except the last one
            # And so on
            # Otherwise, we just use forward backpropagation everywhere
            layers, kwargs = [], []
            forward_simulation_passes = self.forward_simulation_passes or self.n_layers + 1
            for i in range(self.n_layers):
                if self.n_layers - i >= forward_simulation_passes:
                    # Standard layer with spatial backprop
                    layers.append(StandardLayer)
                    kwargs.append(
                        {
                            "cell_type": self.cell_type,
                            "dtype": self.dtype,
                            "unroll": self.unroll,
                            "stop_gradients": "spatial",
                            "name": f"StandardLayer_{i}",
                        }
                    )
                else:
                    layers.append(ForwardBPTTLayer)
                    kwargs.append(
                        {
                            "cell": partial(
                                ForwardBPTTCell,
                                cell_type=self.cell_type,
                                base_precision=self.base_precision,
                                increased_precision=self.increased_precision,
                                approx_inverse=self.approx_inverse,
                                norm_before_readout=self.norm_before_readout,
                                pooling=self.pooling,
                            ),
                            "base_precision": self.base_precision,
                            "increased_precision": self.increased_precision,
                            "two_passes": self.training_mode == "forward_forward",
                            "unroll": self.unroll,
                            "length": x.shape[0],
                            "name": f"ForwardBPTTLayer_{i}",
                        }
                    )
        else:
            raise ValueError(f"Unknown training mode: {self.training_mode}")

        layers = [
            layer(
                hidden_dim=self.hidden_dim,
                output_dim=self.hidden_dim if i < self.n_layers - 1 else self.output_dim,
                **kwarg,
            )
            for i, (layer, kwarg) in enumerate(zip(layers, kwargs))
        ]

        out = x
        for i, layer in enumerate(layers):
            out = layer(out)["output"]

        if self.pooling == "cumulative_mean":
            out = cumulative_mean_pooling(out)
        elif self.pooling == "mean":
            out = (
                jnp.cumsum(out, axis=0) / jnp.arange(1, out.shape[0] + 1)[:, None]
            )  # cumulative mean without stop gradient. NOTE: non-causal in the backward pass
        elif self.pooling != "none":
            raise ValueError(f"Unknown pooling type: {self.pooling}")

        return out
