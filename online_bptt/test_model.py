import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn
from functools import partial
import pytest

from .model import (
    ForwardBPTTCell,
    ForwardBPTTRNN,
    StandardRNN,
    GRUCell,
)
from .utils import check_grad_all

dtype = jnp.float64
jax.config.update("jax_enable_x64", True)


def mse_loss(pred, target, mask):
    return jnp.mean(mask * (pred - target) ** 2)


full_loss = lambda x, y, m: jax.vmap(mse_loss)(x, y, m).sum()


@pytest.fixture
def setup_data():
    key = random.PRNGKey(0)
    input_dim = 1
    hidden_dim = 10
    output_dim = 1
    seq_len = 5
    return key, input_dim, hidden_dim, output_dim, seq_len


def test_forward_bptt_cell_single_step(setup_data):
    key, input_dim, hidden_dim, output_dim, _ = setup_data

    dummy_x = jnp.ones((input_dim,))
    dummy_y = jnp.ones((output_dim,))
    dummy_m = jnp.ones((1,))
    dummy_batch = (dummy_x, dummy_y, dummy_m)

    forward_bptt_cell = ForwardBPTTCell(
        hidden_dim=hidden_dim, output_dim=output_dim, loss_fn=mse_loss
    )
    initial_carry = forward_bptt_cell.initialize_carry(key, (input_dim,))

    params = forward_bptt_cell.init(key, initial_carry, dummy_batch)

    new_carry, out = forward_bptt_cell.apply(params, initial_carry, dummy_batch)

    new_h, new_delta, new_inst_delta, new_prod_jac = new_carry

    assert new_h.shape == (hidden_dim,)
    assert new_delta.shape == (hidden_dim,)
    assert new_inst_delta.shape == (hidden_dim,)
    assert new_prod_jac.shape == (hidden_dim, hidden_dim)
    assert "output" in out


def test_forward_bptt_rnn_sequence(setup_data):
    key, input_dim, hidden_dim, output_dim, seq_len = setup_data
    key = random.PRNGKey(42)

    dummy_inputs = jnp.ones((seq_len, input_dim))
    dummy_targets = jnp.ones((seq_len, output_dim))
    dummy_mask = jnp.ones((seq_len,)).at[: seq_len // 2].set(0.0)  # Half of the sequence is masked
    dummy_batch = {"input": dummy_inputs, "target": dummy_targets, "mask": dummy_mask}

    forward_bptt_rnn = ForwardBPTTRNN(
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        cell=partial(ForwardBPTTCell, loss_fn=mse_loss),
    )

    params = forward_bptt_rnn.init(key, dummy_batch)

    outputs = forward_bptt_rnn.apply(
        params,
        dummy_batch,
    )

    assert outputs["output"].shape == (seq_len, output_dim)


def test_forward_bptt_rnn_backward_pass(setup_data):
    key, input_dim, hidden_dim, output_dim, seq_len = setup_data
    key = random.PRNGKey(42)

    dummy_inputs = jnp.ones((seq_len, input_dim), dtype=dtype)
    dummy_targets = jnp.ones((seq_len, output_dim), dtype=dtype)
    dummy_mask = jnp.ones((seq_len,), dtype=dtype)
    dummy_batch = {"input": dummy_inputs, "target": dummy_targets, "mask": dummy_mask}

    forward_bptt_rnn = ForwardBPTTRNN(
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        cell=partial(ForwardBPTTCell, loss_fn=mse_loss, dtype=dtype),
        dtype=dtype,
    )

    params = forward_bptt_rnn.init(key, dummy_batch)

    standard_rnn = StandardRNN(hidden_dim=hidden_dim, output_dim=output_dim, dtype=dtype)
    standard_params = standard_rnn.init(jax.random.PRNGKey(0), dummy_inputs)

    # Manually copy the weights to compare the same model
    gru_cell_params = params["params"]["ScanForwardBPTTCell_0"]["GRUCell_0"]
    standard_params["params"]["ScanGRUCell_0"] = gru_cell_params

    def standard_loss_fn(p, ic, b):
        y_hat = standard_rnn.apply({"params": p}, b["input"], init_carry=ic)["output"]
        return full_loss(y_hat, b["target"], b["mask"])

    _, (grad_bptt, _) = jax.value_and_grad(standard_loss_fn, argnums=(0, 1))(
        standard_params["params"],
        jnp.zeros((hidden_dim,)),
        dummy_batch,
    )

    def train_step(p, b):
        def loss_fn(_p):
            y_hat = forward_bptt_rnn.apply(_p, b)["output"]
            return full_loss(y_hat, b["target"], b["mask"])

        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(p)
        return grads

    grads = train_step(params, dummy_batch)

    check_grad_all(
        grads["params"]["ScanForwardBPTTCell_0"]["GRUCell_0"],
        grad_bptt["ScanGRUCell_0"],
        rtol=1e-3,
    )
