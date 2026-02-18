"""
Forward backpropagation-through-time playground.
"""

import jax
import flax.linen as nn
from typing import Callable, Any
from functools import partial
from omegaconf import DictConfig
from src.model.cells import (
    GRUCell,
    LRUCell,
)
from src.model.network import RNN


def parameter_conversion_normal_to_forward(params, example_params):
    if "params" in params.keys():
        return {
            "params": parameter_conversion_normal_to_forward(
                params["params"], example_params["params"]
            )
        }

    converted_params = {}
    example_keys = list(example_params.keys())
    for (key, value) in params.items():
        new_key = [k for k in example_keys if k.endswith(f"_{key.split('_')[-1]}")]
        assert len(new_key) == 1
        if new_key[0].startswith("ForwardBPTTLayer"):
            converted_value = {"layer": {"cell": value["layer"]}}
        else:
            converted_value = value
        converted_params[new_key[0]] = converted_value

    return converted_params


def create_model(
    cfg: DictConfig,
    output_dim: int,
    seq_len: int,
    base_precision: Any,
    increased_precision: Any,
    batch: dict,
    key: jax.random.PRNGKey,
):
    """
    Helper function to create a model and convert parameters if needed.
    """
    assert cfg.model.training_mode in ["normal", "spatial", "forward", "forward_forward"]

    if cfg.model.cell == "gru":
        cell_type = partial(
            GRUCell,
            T_min=seq_len * cfg.model.T_min_frac if cfg.model.T_min_frac is not None else None,
            T_max=seq_len * cfg.model.T_max_frac if cfg.model.T_max_frac is not None else None,
            norm_before_readout=cfg.model.norm_before_readout,
            freeze_recurrence=cfg.model.freeze_recurrence,
            dtype=base_precision,
        )
    elif cfg.model.cell == "lru":
        cell_type = partial(
            LRUCell,
            r_min=cfg.model.lru_r_min,
            r_max=cfg.model.lru_r_max,
            norm_before_readout=cfg.model.norm_before_readout,
            freeze_recurrence=cfg.model.freeze_recurrence,
            dtype=base_precision,
        )
    else:
        raise ValueError(f"Unknown cell type: {cfg.model.cell}")

    model = partial(
        RNN,
        hidden_dim=cfg.model.hidden_dim,
        output_dim=output_dim,
        cell_type=cell_type,
        n_layers=cfg.model.n_layers,
        pooling=cfg.model.pooling,
        dtype=base_precision,
        unroll=cfg.model.unroll,
        base_precision=base_precision,
        increased_precision=increased_precision,
        two_passes=cfg.model.training_mode == "forward_forward",
        approx_inverse=cfg.model.approx_inverse,
        norm_before_readout=cfg.model.norm_before_readout,
        forward_simulation_passes=cfg.model.forward_simulation_passes,
    )

    # Always create a model trained in normal mode, to directly use it, or to use its params as ref
    training_mode_ref = (
        "normal"
        if cfg.model.training_mode in ["forward", "forward_forward"]
        else cfg.model.training_mode
    )
    BatchedRNN = nn.vmap(
        partial(model, training_mode=training_mode_ref),
        in_axes=0,
        out_axes=0,
        variable_axes={"params": None},
        split_rngs={"params": False},
    )
    batched_model = BatchedRNN()
    params = batched_model.init(key, batch["input"])["params"]

    if cfg.model.training_mode in ["forward", "forward_forward"]:
        BatchedRNN = nn.vmap(
            partial(model, training_mode=cfg.model.training_mode),
            in_axes=0,
            out_axes=0,
            variable_axes={"params": None},
            split_rngs={"params": False},
        )
        batched_model = BatchedRNN()
        example_params = batched_model.init(key, batch["input"])["params"]
        params = parameter_conversion_normal_to_forward(params, example_params)

    return params, batched_model
