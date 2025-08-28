import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn
from functools import partial
import pytest

from .model.network import (
    ForwardBPTTCell,
    ForwardBPTTRNN,
    StandardRNN,
)
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

    dummy_x = jnp.ones((input_dim,))
    dummy_y = jnp.ones((output_dim,))
    dummy_m = jnp.ones((1,))
    dummy_batch = (dummy_x, dummy_y, dummy_m)

    forward_bptt_cell = ForwardBPTTCell(
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        loss_fn=mse_loss,
        cell_type=cell_class,
    )
    initial_carry = forward_bptt_cell.initialize_carry(
        key, (input_dim,), dtype=jnp.float32, diagonal_jacobian=isinstance(cell_class, LRUCell)
    )

    params = forward_bptt_cell.init(key, initial_carry, dummy_batch)

    new_carry, out = forward_bptt_cell.apply(params, initial_carry, dummy_batch)

    new_h, new_delta, new_inst_delta, new_prod_jac, new_mean, new_t = new_carry

    assert new_h.shape == (hidden_dim,)
    assert new_delta.shape == (hidden_dim,)
    assert new_inst_delta.shape == (hidden_dim,)
    assert new_prod_jac.shape == (hidden_dim, hidden_dim)
    assert new_mean.shape == (output_dim,)
    assert "output" in out


@pytest.mark.parametrize("cell_class", [GRUCell, LRUCell, EUNNCell])
def test_forward_bptt_rnn_sequence(setup_data, cell_class):
    key, input_dim, hidden_dim, output_dim, seq_len = setup_data
    key = random.PRNGKey(42)

    dummy_inputs = jnp.ones((seq_len, input_dim))
    dummy_targets = jnp.ones((seq_len, output_dim))
    dummy_mask = jnp.ones((seq_len,)).at[: seq_len // 2].set(0.0)  # Half of the sequence is masked
    dummy_batch = {"input": dummy_inputs, "target": dummy_targets, "mask": dummy_mask}

    forward_bptt_rnn = ForwardBPTTRNN(
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        cell=partial(ForwardBPTTCell, loss_fn=mse_loss, cell_type=cell_class),
    )

    params = forward_bptt_rnn.init(key, dummy_batch)

    outputs = forward_bptt_rnn.apply(
        params,
        dummy_batch,
    )

    assert outputs["output"].shape == (seq_len, output_dim)


@pytest.mark.parametrize("cell_class", [GRUCell, LRUCell, EUNNCell])
def test_forward_bptt_rnn_backward_pass(setup_data, cell_class):
    key, input_dim, hidden_dim, output_dim, seq_len = setup_data
    key = random.PRNGKey(42)

    dummy_inputs = jnp.ones((seq_len, input_dim), dtype=dtype)
    dummy_targets = jnp.ones((seq_len, output_dim), dtype=dtype)
    dummy_mask = jnp.ones((seq_len,), dtype=dtype)
    dummy_batch = {"input": dummy_inputs, "target": dummy_targets, "mask": dummy_mask}

    pooling = "cumulative_mean"  # NOTE: important to test pooling as it can change quite a bit gradient computation
    forward_bptt_rnn = ForwardBPTTRNN(
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        cell=partial(
            ForwardBPTTCell,
            loss_fn=mse_loss,
            increased_precision=dtype,
            base_precision=dtype,
            cell_type=cell_class,
        ),
        increased_precision=dtype,
        base_precision=dtype,
        pooling=pooling,
    )

    params = forward_bptt_rnn.init(key, dummy_batch)

    standard_rnn = StandardRNN(
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        dtype=dtype,
        cell_type=cell_class,
        pooling=pooling,
    )
    standard_params = standard_rnn.init(jax.random.PRNGKey(0), dummy_batch)

    # Manually copy the weights to compare the same model
    cell_params = params["params"]["ScanForwardBPTTCell_0"]["cell"]
    cell_name = list(standard_params["params"].keys())[0]
    standard_params["params"][cell_name] = cell_params

    def standard_loss_fn(p, ic, b):
        y_hat = standard_rnn.apply({"params": p}, b, init_carry=ic)["output"]
        return full_loss(y_hat, b["target"], b["mask"])

    loss_bptt, (grad_bptt, _) = jax.value_and_grad(standard_loss_fn, argnums=(0, 1))(
        standard_params["params"],
        standard_rnn.apply(
            standard_params,
            jax.random.PRNGKey(0),
            (input_dim,),
            method=standard_rnn.initialize_carry,
        ),
        dummy_batch,
    )

    def train_step(p, b):
        def loss_fn(_p):
            y_hat = forward_bptt_rnn.apply(_p, b)["output"]
            return full_loss(y_hat, b["target"], b["mask"])

        return jax.value_and_grad(loss_fn)(p)

    loss, grads = train_step(params, dummy_batch)

    assert jnp.allclose(loss, loss_bptt)

    cell_grad_name = list(grad_bptt.keys())[0]
    check_grad_all(
        grads["params"]["ScanForwardBPTTCell_0"]["cell"],
        grad_bptt[cell_grad_name],
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
