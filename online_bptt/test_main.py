import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn
from functools import partial
import pytest

from .main import (
    ForwardBPTTCell,
    ForwardBPTTRNN,
    StandardRNN,
    GRUCell,
)
from .utils import check_grad_all


print(jax.__version__)

def mse_loss(y_hat, y):
    return jnp.mean((y_hat - y) ** 2)


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

    forward_bptt_cell = ForwardBPTTCell(
        hidden_dim=hidden_dim, output_dim=output_dim, loss_fn=mse_loss
    )
    initial_carry = forward_bptt_cell.initialize_carry(key, (input_dim,))

    params = forward_bptt_cell.init(key, initial_carry, (dummy_x, dummy_y))

    new_carry, out = forward_bptt_cell.apply(params, initial_carry, (dummy_x, dummy_y))

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

    forward_bptt_rnn = ForwardBPTTRNN(
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        cell=partial(ForwardBPTTCell, loss_fn=mse_loss),
    )

    params = forward_bptt_rnn.init(key, dummy_inputs, dummy_targets)

    outputs = forward_bptt_rnn.apply(
        params,
        dummy_inputs,
        dummy_targets,
    )

    assert outputs.shape == (seq_len, output_dim)


def test_forward_bptt_rnn_backward_pass(setup_data):
    key, input_dim, hidden_dim, output_dim, seq_len = setup_data
    key = random.PRNGKey(42)

    dummy_inputs = jnp.ones((seq_len, input_dim))
    dummy_targets = jnp.ones((seq_len, output_dim))

    forward_bptt_rnn = ForwardBPTTRNN(
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        cell=partial(ForwardBPTTCell, loss_fn=mse_loss),
    )

    params = forward_bptt_rnn.init(key, dummy_inputs, dummy_targets)

    standard_rnn = StandardRNN(hidden_dim=hidden_dim, output_dim=output_dim)
    standard_params = standard_rnn.init(jax.random.PRNGKey(0), dummy_inputs)

    # Manually copy the weights to compare the same model
    gru_cell_params = params["params"]["ScanForwardBPTTCell_0"]["GRUCell_0"]
    standard_params["params"]["ScanGRUCell_0"] = gru_cell_params

    def standard_loss_fn(p, ic, inputs, targets):
        y_hat = standard_rnn.apply({"params": p}, inputs, init_carry=ic)
        return mse_loss(y_hat, targets) * seq_len

    _, (grad_bptt, _) = jax.value_and_grad(standard_loss_fn, argnums=(0, 1))(
        standard_params["params"], jnp.zeros((hidden_dim,)), dummy_inputs, dummy_targets
    )

    def train_step(p, inputs, targets):
        def loss_fn(_p):
            y_hat = forward_bptt_rnn.apply(_p, inputs, targets)
            return jnp.mean((y_hat - targets) ** 2) * seq_len

        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(p)
        return grads

    grads = train_step(params, dummy_inputs, dummy_targets)

    check_grad_all(
        grads["params"]["ScanForwardBPTTCell_0"]["GRUCell_0"],
        grad_bptt["ScanGRUCell_0"],
        rtol=1e-3,
    )
