import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn
from functools import partial
import pytest

from .model.network import (
    ForwardBPTTCell,
    RNN,
)
from .model_factory import parameter_conversion_normal_to_forward
from .model.cells import GRUCell, EUNNCell, LRUCell
from .utils import check_grad_all

dtype = jnp.float64
# Run tests on CPU to avoid GPU OOM/solver issues in CI
jax.config.update("jax_platform_name", "cpu")
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


@pytest.mark.parametrize("cell_class", [GRUCell, LRUCell, EUNNCell])
def test_forward_bptt_cell_single_step(setup_data, cell_class):
    key, input_dim, hidden_dim, output_dim, _ = setup_data

    dummy_x = jnp.ones((input_dim,), dtype=jnp.float32)
    dummy_y = jnp.ones((output_dim,), dtype=jnp.float32)

    forward_bptt_cell = ForwardBPTTCell(
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        cell_type=cell_class,
    )
    initial_carry = forward_bptt_cell.initialize_carry(
        key, (input_dim,), dtype=jnp.float32, diagonal_jacobian=isinstance(cell_class, LRUCell)
    )

    params = forward_bptt_cell.init(key, initial_carry, (dummy_x, dummy_y))

    new_carry, out = forward_bptt_cell.apply(params, initial_carry, (dummy_x, dummy_y))

    new_h, new_delta, new_inst_delta, new_prod_jac = new_carry

    assert new_h.shape == (hidden_dim,)
    assert new_delta.shape == (hidden_dim,)
    assert new_inst_delta.shape == (hidden_dim,)
    assert new_prod_jac.shape == (hidden_dim, hidden_dim)
    assert "output" in out


@pytest.mark.parametrize("cell_class", [GRUCell, LRUCell, EUNNCell])
def test_forward_bptt_rnn_sequence(setup_data, cell_class):
    key, input_dim, hidden_dim, output_dim, seq_len = setup_data
    key = random.PRNGKey(42)

    dummy_inputs = jnp.ones((seq_len, input_dim))

    forward_bptt_rnn = RNN(
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        training_mode="forward_forward",
        cell_type=cell_class,
    )

    params = forward_bptt_rnn.init(key, dummy_inputs)

    outputs = forward_bptt_rnn.apply(params, dummy_inputs)

    assert outputs.shape == (seq_len, output_dim)


@pytest.mark.parametrize("cell_class", [GRUCell, LRUCell, EUNNCell])
def test_forward_bptt_rnn_backward_pass(setup_data, cell_class):
    key, input_dim, hidden_dim, output_dim, seq_len = setup_data
    key = random.PRNGKey(42)

    dummy_inputs = jnp.ones((seq_len, input_dim), dtype=dtype)
    dummy_targets = jnp.ones((seq_len, output_dim), dtype=dtype)
    dummy_mask = jnp.ones((seq_len,), dtype=dtype)
    dummy_batch = {"input": dummy_inputs, "target": dummy_targets, "mask": dummy_mask}

    pooling = "cumulative_mean"  # NOTE: important to test pooling as it can change quite a bit gradient computation
    rnn_func = partial(
        RNN,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        increased_precision=dtype,
        base_precision=dtype,
        cell_type=cell_class,
        pooling=pooling,
    )
    forward_bptt_rnn = rnn_func(training_mode="forward_forward")
    standard_rnn = rnn_func(training_mode="normal")

    standard_params = standard_rnn.init(key, dummy_inputs)
    params = parameter_conversion_normal_to_forward(standard_params)

    def loss_fn(model, p, b):
        y_hat = model.apply(p, b["input"])
        return full_loss(y_hat, b["target"], b["mask"])

    loss_bptt, grad_bptt = jax.value_and_grad(loss_fn, argnums=(1))(
        standard_rnn, standard_params, dummy_batch
    )

    loss, grads = jax.value_and_grad(loss_fn, argnums=(1))(
        forward_bptt_rnn, params, dummy_batch
    )

    assert jnp.allclose(loss, loss_bptt)
    check_grad_all(
        grads["params"]["ForwardBPTTLayer_0"]["layer"]["cell"],
        grad_bptt["params"]["StandardLayer_0"]["layer"],
        rtol=1e-3,
    )


def test_eunn_perm_preserves_norm_one_step_no_input():
    key = random.PRNGKey(0)
    input_dim = 4
    hidden_dim = 8  # even for clean pairing
    output_dim = 1

    # Build the cell (float64 for tighter numeric tolerance)
    cell = EUNNCell(
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        n_layers=4,
        dtype=dtype,
    )

    # Initialize parameters for the specific method to avoid running full __call__
    v0 = jnp.zeros((hidden_dim,), dtype=jnp.complex128)
    params = cell.init(key, v0, method=EUNNCell._apply_layers_vec)

    # Random complex hidden state
    k1, k2 = random.split(key)
    h_real = random.normal(k1, (hidden_dim,), dtype=dtype)
    h_imag = random.normal(k2, (hidden_dim,), dtype=dtype)
    h = (h_real + 1j * h_imag).astype(jnp.complex128)

    # Apply one unitary recurrence step without input contribution via internal vector transform
    h_next = cell.apply(params, h, method=EUNNCell._apply_layers_vec)

    # Norm should be preserved
    norm_h = jnp.linalg.norm(h)
    norm_h_next = jnp.linalg.norm(h_next)
    print(norm_h, norm_h_next)
    assert jnp.allclose(norm_h_next, norm_h, rtol=1e-9, atol=1e-9)
