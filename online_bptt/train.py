import hydra
from omegaconf import DictConfig
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
import flax.linen as nn
from tqdm import tqdm
from functools import partial

from online_bptt.data import create_dataloader
from online_bptt.model import ForwardBPTTCell, StandardRNN, ForwardBPTTRNN, GRUCell


@partial(jax.jit, static_argnums=(0, 1, 2, 3))
def train_step(model, training_mode, loss_fn, acc_fn, state, batch):
    def _loss_fn(params):
        if training_mode == "normal":
            out = model.apply({"params": params}, batch["input"])
        else:
            out = model.apply({"params": params}, batch)

        pred = out.pop("output")
        loss = loss_fn(pred, batch["target"], batch["mask"])
        acc = acc_fn(pred, batch["target"], batch["mask"])
        return loss, {**out, "acc": acc}

    (loss, extra), grads = jax.value_and_grad(_loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, extra


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    key = jax.random.PRNGKey(cfg.seed)

    # Dataloader
    if cfg.model.precision == "float64":
        jax.config.update("jax_enable_x64", True)
        dtype = jnp.float64
    else:
        dtype = jnp.float32

    # Create dataloader
    dataloader, loss_fn, accuracy_fn = create_dataloader(
        task=cfg.data.task,
        batch_size=cfg.data.batch_size,
        n_samples=cfg.data.n_samples,
        seed=cfg.seed,
        seq_len=cfg.data.seq_len,
        bit_width=cfg.data.bit_width,
        waiting_time=cfg.data.waiting_time,
    )
    dummy_batch = next(iter(dataloader))
    dummy_batch = jax.tree.map(lambda x: x.astype(dtype), dummy_batch)
    output_dim = dummy_batch["target"].shape[-1]
    seq_len = dummy_batch["input"].shape[-2]
    n_train_steps = len(dataloader)

    full_loss_fn = lambda x, y, m: jax.vmap(jax.vmap(loss_fn))(x, y, m).sum()
    full_accuracy_fn = lambda x, y, m: jax.vmap(jax.vmap(accuracy_fn))(x, y, m).sum() / m.sum()

    # Instantiate model
    cell_type = partial(GRUCell, T_min=seq_len, T_max=seq_len*2, dtype=dtype)
    if cfg.model.training_mode == "normal":
        BatchedRNN = nn.vmap(
            partial(StandardRNN, cell_type=cell_type, dtype=dtype),
            in_axes=0,
            out_axes=0,
            variable_axes={"params": None},
            split_rngs={"params": False},
        )
    elif cfg.model.training_mode == "forward_bptt":

        model = partial(
            ForwardBPTTRNN,
            cell=partial(ForwardBPTTCell, cell_type=cell_type, loss_fn=loss_fn, dtype=dtype),
            dtype=dtype,
        )
        BatchedRNN = nn.vmap(
            model,
            in_axes=0,
            out_axes=0,
            variable_axes={"params": None},
            split_rngs={"params": False},
        )
    else:
        raise ValueError(f"Unknown training_mode: {cfg.model.training_mode}")

    batched_model = BatchedRNN(hidden_dim=cfg.model.hidden_dim, output_dim=output_dim, dtype=dtype)

    # Initialize parameters
    key, init_key = jax.random.split(key)
    if cfg.model.training_mode == "normal":
        params = batched_model.init(init_key, dummy_batch["input"])["params"]
    else:
        params = batched_model.init(init_key, dummy_batch)["params"]

    # Create optimizer and state
    if cfg.training.scheduler == "cosine":
        lr = optax.cosine_decay_schedule(
            init_value=cfg.training.learning_rate,
            decay_steps=n_train_steps,
        )
    elif cfg.training.scheduler == "constant":
        lr = cfg.learning_rate
    else:
        raise ValueError(f"Unknown scheduler: {cfg.training.scheduler}")
    optimizer = optax.adam(lr)

    state = train_state.TrainState.create(apply_fn=batched_model.apply, params=params, tx=optimizer)

    # Print the type of the different things
    print("Input type:", dummy_batch["input"].dtype)
    print("Target type:", dummy_batch["target"].dtype)
    print("Mask type:", dummy_batch["mask"].dtype)
    print("Model parameters type:", jax.tree.map(lambda x: x.dtype, state.params))

    # Training loop
    pbar = tqdm(range(n_train_steps))
    for step in pbar:
        batch = next(iter(dataloader))
        batch = jax.tree.map(lambda x: x.astype(dtype), batch)
        state, loss, extra = train_step(
            batched_model, cfg.model.training_mode, full_loss_fn, full_accuracy_fn, state, batch
        )

        if step % cfg.training.log_every_steps == 0:
            txt = f"Step {step}, Loss: {loss:.4f}, Accuracy: {extra['acc']:.4f}"
            if "norm_prod_jac" in extra:
                txt += f", Jac: {jnp.mean(extra['norm_prod_jac']):.4f}"
            if "residual_error_delta" in extra:
                txt += f", Res: {jnp.mean(extra['residual_error_delta']):.4f}"
            if "norm_delta_0" in extra:
                txt += f", d0: {jnp.mean(extra['norm_delta_0']):.4f}"
            pbar.set_description(txt)


if __name__ == "__main__":
    main()
