# Online-BPTT

This repository accompanies the blog post [Forward Propagation of Errors Through Time](https://nzucchet.github.io/Online-BPTT/). We investigate whether backpropagation through time (BPTT) can be replaced by an exact forward-mode algorithm for training recurrent neural networks. The code implements the Forward Propagation of Errors Through Time (FPTT) algorithm and reproduces all experiments described in the post.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. Dependencies are platform-conditional: CUDA on Linux, Metal on macOS.

### Linux (CUDA)

```bash
pip install uv
uv venv env && source env/bin/activate
uv pip install -e .
```

### macOS (Metal)

```bash
pip install uv
uv venv env && source env/bin/activate
uv pip install -e .
```

Dependencies are automatically resolved based on your platform via environment markers in `pyproject.toml`.

## Quick Start

Train a model on the copy task (default config):

```bash
python src/train.py training.wandb_log=false
```

Train with forward propagation of errors on downsampled MNIST:

```bash
python src/train.py data=mnist98 model.cell=lru model.n_layers=4 \
  model.hidden_dim=32 model.lru_r_min=0.99 model.pooling=cumulative_mean \
  model.training_mode=forward_forward training.wandb_log=false
```

### Test Configurations

Small-scale test configs are provided for quick validation (`n_samples=64`, `batch_size=4`):

```bash
# Run all experiments with tiny datasets
bash scripts/test_experiments.sh
```

Available test data configs: `copy_test`, `mnist98_test`, `mnist_test`.

### Key Config Options

| Option | Values | Description |
|--------|--------|-------------|
| `model.training_mode` | `normal`, `spatial`, `forward`, `forward_forward` | Training algorithm |
| `model.cell` | `gru`, `lru` | Recurrent cell type |
| `model.n_layers` | int | Number of recurrent layers |
| `model.hidden_dim` | int | Hidden state dimension |
| `model.pooling` | `none`, `cumulative_mean` | Pooling strategy |
| `model.freeze_recurrence` | bool | Freeze recurrent parameters |
| `training.wandb_log` | bool | Enable W&B logging |

## Running Tests

```bash
uv run pytest src/test_model.py -v
```

## Project Structure

```
Online-BPTT/
├── src/
│   ├── model/
│   │   ├── cells.py          # LRUCell, GRUCell implementations
│   │   └── network.py        # RNN, ForwardBPTTCell
│   ├── conf/
│   │   ├── config.yaml       # Default config
│   │   └── data/             # Data configs (copy, mnist, mnist98 + test variants)
│   ├── train.py              # Training script (Hydra)
│   ├── model_factory.py      # Model creation and param conversion
│   ├── test_model.py         # Unit tests
│   ├── data.py               # Data loading
│   ├── metrics.py            # Loss and accuracy functions
│   └── utils.py              # Utilities
├── analysis/                 # Jupyter notebooks for result analysis
├── scripts/
│   └── test_experiments.sh   # Run all experiments with tiny datasets
├── docs/                     # Blog post (GitHub Pages)
└── pyproject.toml
```

## Blog Post

The accompanying blog post is hosted at [nzucchet.github.io/Online-BPTT](https://nzucchet.github.io/Online-BPTT/) and details the derivation of the FPTT algorithm, experimental results, and analysis of its numerical stability limitations.

## Citation

```bibtex
@misc{zucchet2026fptt,
  title={Forward propagation of errors through time},
  author={Zucchet, Nicolas and Pourcel, Guillaume and Ernoult, Maxence},
  year={2026},
  url={https://nzucchet.github.io/Online-BPTT/}
}
```

## License

TBD
