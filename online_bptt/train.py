import hydra
from omegaconf import DictConfig, OmegaConf
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
import flax.linen as nn
from tqdm import tqdm
from functools import partial
import wandb

from online_bptt.data import create_dataloader
from online_bptt.model import (
    ForwardBPTTCell,
    StandardRNN,
    ForwardBPTTRNN,
    GRUCell,
    conversion_params_normal_to_forwardbptt,
)
from online_bptt import metrics


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

    wandb.init(
        project="online-bptt",
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
    )

    # Dataloader
    if cfg.model.precision == "float64":
        jax.config.update("jax_enable_x64", True)
        dtype = jnp.float64
    else:
        dtype = jnp.float32

    # Create dataloader
    dataloader = create_dataloader(
        task=cfg.data.task,
        batch_size=cfg.data.batch_size,
        n_samples=cfg.data.n_samples,
        seed=cfg.seed,
        **{k: v for k, v in cfg.data.items() if k not in ["task", "batch_size", "n_samples"]},
    )
    full_accuracy_fn, loss_fn = metrics.select_metrics(
        cfg.data.classification,
        cfg.data.multiple_pred_per_timestep,
        cfg.data.dense_prediction,
    )
    dummy_batch = next(iter(dataloader))
    dummy_batch = jax.tree.map(lambda x: x.astype(dtype), dummy_batch)
    output_dim = cfg.data.n_classes if cfg.data.classification else cfg.data.output_dim
    seq_len = dummy_batch["input"].shape[-2]
    n_train_steps = len(dataloader)

    full_loss_fn = lambda x, y, m: jax.vmap(jax.vmap(loss_fn))(x, y, m).sum() / m.sum()

    # Instantiate model
    assert cfg.model.training_mode in ["normal", "forward", "forward_forward"]
    cell_type = partial(
        GRUCell,
        T_min=seq_len * cfg.model.T_min_frac if cfg.model.T_min_frac is not None else None,
        T_max=seq_len * cfg.model.T_max_frac if cfg.model.T_max_frac is not None else None,
        dtype=dtype,
    )  # Long time scales to give forward BPTT a chance
    BatchedRNN = nn.vmap(
        partial(StandardRNN, cell_type=cell_type, dtype=dtype),
        in_axes=0,
        out_axes=0,
        variable_axes={"params": None},
        split_rngs={"params": False},
    )
    key, init_key = jax.random.split(key)
    batched_model = BatchedRNN(hidden_dim=cfg.model.hidden_dim, output_dim=output_dim, dtype=dtype)
    params = batched_model.init(init_key, dummy_batch["input"])["params"]

    if cfg.model.training_mode in ["forward", "forward_forward"]:
        # Overwrite the model to use the correct one, and convert parameters
        model = partial(
            ForwardBPTTRNN,
            cell=partial(ForwardBPTTCell, cell_type=cell_type, loss_fn=loss_fn, dtype=dtype),
            dtype=dtype,
            two_passes=cfg.model.training_mode == "forward_forward",
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
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.training.gradient_clipping),
        optax.adam(lr),
    )

    state = train_state.TrainState.create(apply_fn=batched_model.apply, params=params, tx=optimizer)

    # Training loop
    pbar = tqdm(range(n_train_steps))
    for step in pbar:
        batch = next(iter(dataloader))
        batch = jax.tree.map(lambda x: x.astype(dtype), batch)
        state, loss, extra = train_step(
            batched_model, cfg.model.training_mode, full_loss_fn, full_accuracy_fn, state, batch
        )

        if step % cfg.training.log_every_steps == 0:
            log_dict = {**extra, "loss": loss}
            txt = f"Step {step}, Loss: {loss:.4f}, Accuracy: {log_dict['acc']:.4f}"
            pbar.set_description(txt)
            wandb.log(jax.tree.map(jnp.mean, log_dict), step=step)


if __name__ == "__main__":
    main()
