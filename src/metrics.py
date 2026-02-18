import jax
import jax.numpy as jnp


def mse_loss(pred, target, mask):
    return mask * jnp.mean((pred - target) ** 2)


def multi_bit_bce_loss(pred, target, mask):
    # pred: [N], target: [N], {0, 1}, mask: []
    # Binary cross-entropy with logits
    bce = target * jax.nn.log_sigmoid(pred) + (1 - target) * jax.nn.log_sigmoid(-pred)
    return -jnp.sum(mask * bce)


def cross_entropy_loss(pred, target, mask):
    # pred: [C], target: [], mask: []
    ce = -jnp.sum(
        jax.nn.one_hot(target, num_classes=pred.shape[-1]) * jax.nn.log_softmax(pred), axis=-1
    )
    out = jnp.sum(mask * ce)
    return out


def accuracy(pred, target, mask):
    # pred: [N], target: [N], mask: []
    pred_labels = jnp.argmax(pred, axis=-1)  # Get predicted labels from logits
    correct = jnp.sum(mask * (pred_labels == target))
    return correct


def multi_bit_accuracy(pred, target, mask):
    # pred: [N], target: [N], {0, 1}, mask: []
    pred_labels = (pred > 0).astype(jnp.float32)  # Get binary predictions from logits
    correct = jnp.sum(mask * (pred_labels == target))
    return correct / target.shape[-1]


def select_metrics(classification, multiple_pred_per_timestep, dense_prediction):
    if classification:
        if multiple_pred_per_timestep:
            loss_fn = multi_bit_bce_loss
            core_accuracy_fn = multi_bit_accuracy
        else:
            loss_fn = cross_entropy_loss
            core_accuracy_fn = accuracy
        if dense_prediction:
            # Average over all time steps and batches
            accuracy_fn = (
                lambda x, y, m: jax.vmap(jax.vmap(core_accuracy_fn))(x, y, m).sum() / m.sum()
            )
        else:
            # Just return the accuracy for the last time step
            accuracy_fn = (
                lambda x, y, m: jax.vmap(core_accuracy_fn)(x[:, -1], y[:, -1], m[:, -1]).sum()
                / m[:, -1].sum()
            )

    else:
        accuracy_fn = lambda x, y, m: -1
        loss_fn = mse_loss

    # Returns three functions that take (pred, target, mask) as input
    # accuracy_fn returns adequate accuracy for classification tasks (takes entire sequence as input)
    # loss_fn returns loss per time step for online learning
    # NOTE: it will be aggreated to form the total loss
    return accuracy_fn, loss_fn
