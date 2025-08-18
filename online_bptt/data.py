import jax
import jax.numpy as jnp
from jax import random
import chex
from functools import partial
import tensorflow_datasets as tfds


def sample_copy_task(
    key: chex.PRNGKey,
    seq_len: int = 10,
    bit_width: int = 3,
    waiting_time: int = 0,
    output_dim: int = 3,
):
    """
    Sample one copy task sequence with bit-encoded memorization sequence
    and one-hot encoded delimiter/padding.

    Args:
        key: JAX random key
        seq_len: Length of the sequence to copy
        bit_width: Number of bits per element in sequence to memorize
        waiting_time: Number of padding tokens between delimiter and target
        output_dim: Output dimension (not used in this task, but required for compatibility)

    Returns:
        Dictionary containing:
        - 'input': Input sequence with bit encoding for data, one-hot for special tokens
        - 'target': Target sequence (bits only during target positions)
        - 'mask': Binary mask indicating valid positions for loss computation
    """
    assert output_dim == bit_width

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


class MNISTDataset:
    def __init__(self, split):
        dataset = tfds.load("mnist", split=split)
        self.data = []
        self.labels = []
        for example in dataset:
            self.data.append(example["image"])
            self.labels.append(example["label"])
        self.data = jnp.array(self.data)
        self.labels = jnp.array(self.labels)

        # Flatten inputs
        self.data = self.data.reshape(self.data.shape[0], -1, 1).astype(jnp.float32)

        # Have one label per time step
        self.labels = self.labels.reshape(-1, 1)  # Shape: (N, 1)
        self.labels = jnp.repeat(self.labels, self.data.shape[1], axis=1)  # Shape: (N, T)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        print("data", self.data[idx].shape, self.labels[idx].shape)
        return {
            "input": self.data[idx],
            "target": self.labels[idx],
            "mask": jnp.ones_like(self.data[idx])[:, :, 0],
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


class DatasetDataLoader:
    """Dataloader which iterates for multiple epochs over the dataset."""

    def __init__(self, dataset, batch_size, n_samples, seed):
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.key = random.PRNGKey(seed)

    def __iter__(self):
        i = 0
        while i < self.n_samples:
            # One epoch
            key, self.key = random.split(self.key)
            batch_indices = random.permutation(key, jnp.arange(len(self.dataset)))
            for j in range(0, len(self.dataset), self.batch_size):
                batch = batch_indices[j : j + self.batch_size]
                i += self.batch_size
                yield self.dataset[batch]

    def __len__(self):
        epochs_size = len(self.dataset) // self.batch_size
        n_epochs = jnp.ceil(self.n_samples / len(self.dataset)).astype(int)
        return n_epochs * epochs_size


def create_dataloader(task, batch_size, n_samples, seed, **kwargs):
    if task == "copy":
        _args = {k: kwargs[k] for k in ["seq_len", "bit_width", "waiting_time", "output_dim"]}
        sample_fn = partial(sample_copy_task, **_args)
        dataloader = DataLoader(sample_fn, batch_size, n_samples, seed)
    elif task == "mnist":
        dataset = MNISTDataset(split="train")
        dataloader = DatasetDataLoader(dataset, batch_size, n_samples, seed)
    else:
        raise ValueError(f"Unknown task: {task}")

    return dataloader
