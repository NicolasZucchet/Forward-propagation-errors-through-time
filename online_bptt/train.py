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

from online_bptt.data import create_dataloaders
from online_bptt.model import create_model
from online_bptt import metrics


@partial(jax.jit, static_argnums=(0, 1, 2))
def train_step(model, loss_fn, acc_fn, state, batch):
    def _loss_fn(params):
        out = model.apply({"params": params}, batch)
        pred = out.pop("output")
        loss = loss_fn(pred, batch["target"], batch["mask"])
        acc = acc_fn(pred, batch["target"], batch["mask"])
        return loss, {**out, "acc": acc}

    (loss, extra), grads = jax.value_and_grad(_loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, extra


@partial(jax.jit, static_argnums=(0, 1, 2))
def eval_step(model, loss_fn, acc_fn, state, batch):
    """Evaluates the model on the given dataloader."""
    out = model.apply({"params": state.params}, batch)
    pred = out.pop("output")
    loss = loss_fn(pred, batch["target"], batch["mask"])
    acc = acc_fn(pred, batch["target"], batch["mask"])
    return {"loss": loss, "acc": acc}


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

    # Create dataloaders
    train_loader, val_loader, eval_frequency = create_dataloaders(
        task=cfg.data.task,
        batch_size=cfg.data.batch_size,
        n_samples=cfg.data.n_samples,
        eval_frequency=cfg.training.eval_frequency,
        seed=cfg.seed,
        **{
            k: v
            for k, v in cfg.data.items()
            if k not in ["task", "batch_size", "n_samples", "eval_frequency"]
        },
    )

    full_accuracy_fn, loss_fn = metrics.select_metrics(
        cfg.data.classification,
        cfg.data.multiple_pred_per_timestep,
        cfg.data.dense_prediction,
    )
    dummy_batch = next(iter(train_loader))
    dummy_batch = jax.tree.map(lambda x: x.astype(dtype), dummy_batch)
    output_dim = cfg.data.n_classes if cfg.data.classification else cfg.data.output_dim
    seq_len = dummy_batch["input"].shape[-2]
    n_train_steps = len(train_loader)

    full_loss_fn = lambda x, y, m: jax.vmap(jax.vmap(loss_fn))(x, y, m).sum() / m.sum()

    # Instantiate model
    key, key_model = jax.random.split(key)
    params, model = create_model(cfg, output_dim, seq_len, loss_fn, dtype, dummy_batch, key_model)

    # Create optimizer and state
    if cfg.training.scheduler == "cosine":
        lr = optax.cosine_decay_schedule(
            init_value=cfg.training.learning_rate,
            decay_steps=n_train_steps,
        )
    elif cfg.training.scheduler == "constant":
        lr = cfg.training.learning_rate
    else:
        raise ValueError(f"Unknown scheduler: {cfg.training.scheduler}")
    optimizer_name = cfg.training.get("optimizer", "adam")
    if optimizer_name == "adamw":
        optimizer = optax.chain(
            optax.clip_by_global_norm(cfg.training.gradient_clipping),
            optax.adamw(lr, weight_decay=cfg.training.weight_decay),
        )
    elif optimizer_name == "adam":
        optimizer = optax.chain(
            optax.clip_by_global_norm(cfg.training.gradient_clipping),
            optax.adam(lr),
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

    # Training loop
    pbar = tqdm(range(n_train_steps))
    log_accumulator = None
    for step in pbar:
        batch = next(iter(train_loader))
        batch = jax.tree.map(lambda x: x.astype(dtype), batch)
        state, loss, extra = train_step(model, full_loss_fn, full_accuracy_fn, state, batch)

        current_log = {"loss": loss, **extra}
        if log_accumulator is None:
            log_accumulator = jax.tree.map(jnp.zeros_like, current_log)
        log_accumulator = jax.tree.map(lambda x, y: x + y, log_accumulator, current_log)

        # Log training metrics
        if step % cfg.training.log_every_steps == 0:
            log_dict = jax.tree.map(lambda x: x / cfg.training.log_every_steps, log_accumulator)
            log_dict = jax.tree.map(jnp.mean, log_dict)
            txt = f"Step {step}, Loss: {log_dict['loss']:.4f}, Accuracy: {log_dict['acc']:.4f}"
            pbar.set_description(txt)
            wandb.log(log_dict, step=step)
            log_accumulator = jax.tree.map(jnp.zeros_like, log_accumulator)

        # Eval whenever wanted
        if val_loader is not None and step % eval_frequency == 0:
            print(f"Evaluating at step {step}", end=" ")
            val_metrics = []
            for batch in val_loader:
                val_metrics.append(eval_step(model, full_loss_fn, full_accuracy_fn, state, batch))
            val_metrics = jax.tree.map(lambda *x: jnp.mean(jnp.array(x)), *val_metrics)
            results = ""
            for k, v in val_metrics.items():
                results += f"{k}: {v:.4f}, "
            print(results[:-2])
            wandb.log(val_metrics, step=step)


if __name__ == "__main__":
    main()
