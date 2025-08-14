import jax
import jax.numpy as jnp
from jax import random
import chex
from functools import partial


def sample_copy_task(
    key: chex.PRNGKey,
    seq_len: int = 10,
    bit_width: int = 3,
    waiting_time: int = 0,
):
    """
    Sample one copy task sequence with bit-encoded memorization sequence
    and one-hot encoded delimiter/padding.

    Args:
        key: JAX random key
        seq_len: Length of the sequence to copy
        bit_width: Number of bits per element in sequence to memorize
        waiting_time: Number of padding tokens between delimiter and target

    Returns:
        Dictionary containing:
        - 'input': Input sequence with bit encoding for data, one-hot for special tokens
        - 'target': Target sequence (bits only during target positions)
        - 'mask': Binary mask indicating valid positions for loss computation
    """
    # Sample random bit sequence to copy
    seq_to_copy_bits = random.randint(key, shape=(seq_len, bit_width), minval=0, maxval=2).astype(
        jnp.float32
    )

    # Total sequence length: seq_len + 1 (delimiter) + waiting_time + seq_len (target)
    total_len = 2 * seq_len + 1 + waiting_time
    input_dim = bit_width + 2  # bit_width + delimiter + padding channels

    # Initialize input sequence
    input_seq = jnp.zeros((total_len, input_dim))

    # Set the bit-encoded sequence to memorize (first seq_len positions)
    input_seq = input_seq.at[:seq_len, :bit_width].set(seq_to_copy_bits)

    # Set delimiter (one-hot at position bit_width)
    delimiter_pos = seq_len + waiting_time
    input_seq = input_seq.at[delimiter_pos, bit_width].set(1.0)

    # Set padding tokens (one-hot at position bit_width + 1)
    padding_start = seq_len
    padding_end = seq_len + waiting_time
    input_seq = input_seq.at[padding_start:padding_end, bit_width + 1].set(1.0)

    # Create target sequence: only bits during target positions
    target_seq = jnp.zeros((total_len, bit_width))
    target_start = seq_len + 1 + waiting_time
    target_seq = target_seq.at[target_start : target_start + seq_len].set(seq_to_copy_bits)

    # Create mask: 1s where we want to compute loss (target positions only)
    mask = jnp.zeros(total_len)
    mask = mask.at[target_start : target_start + seq_len].set(1.0)

    return {
        "input": input_seq,  # Shape: (total_len, bit_width + 2)
        "target": target_seq,  # Shape: (total_len, bit_width)
        "mask": mask,  # Shape: (total_len,)
    }


class DataLoader:
    def __init__(self, sample_fn, batch_size, n_samples, seed):
        self.sample_fn = jax.jit(sample_fn)
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.key = random.PRNGKey(seed)

    def __iter__(self):
        for _ in range(self.n_samples // self.batch_size):
            self.key, batch_key = random.split(self.key)
            yield jax.vmap(self.sample_fn)(random.split(batch_key, self.batch_size))

    def __len__(self):
        return self.n_samples // self.batch_size


def mse_loss(pred, target, mask):
    return mask * jnp.mean((pred - target) ** 2)


def multi_bit_bce_loss(pred, target, mask):
    # pred: [N], target: [N], {0, 1}, mask: []
    # Binary cross-entropy with logits
    bce = target * jax.nn.log_sigmoid(pred) + (1 - target) * jax.nn.log_sigmoid(-pred)
    return -jnp.sum(mask * bce)


def multi_bit_accuracy(pred, target, mask):
    # pred: [N], target: [N], {0, 1}, mask: []
    pred_labels = (pred > 0).astype(jnp.float32)  # Get binary predictions from logits
    correct = jnp.sum(mask * (pred_labels == target))
    return correct / target.shape[-1]


def create_dataloader(task, batch_size, n_samples, seed, **kwargs):
    if task == "copy":
        sample_fn = partial(sample_copy_task, **kwargs)
        loss_fn = lambda x, y, m: multi_bit_bce_loss(x, y, m) / kwargs['seq_len']
        accuracy_fn = multi_bit_accuracy
    else:
        raise ValueError(f"Unknown task: {task}")

    return DataLoader(sample_fn, batch_size, n_samples, seed), loss_fn, accuracy_fn
