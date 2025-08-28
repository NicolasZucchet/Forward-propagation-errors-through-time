"""
Forward backpropagation-through-time playground.
"""

import jax
import flax.linen as nn
from typing import Callable, Any
from functools import partial
from omegaconf import DictConfig
from online_bptt.model.cells import (
    GRUCell,
    LRUCell,
    EUNNCell,
)
from online_bptt.model.network import StandardRNN, ForwardBPTTRNN, ForwardBPTTCell


def create_model(
    cfg: DictConfig,
    output_dim: int,
    seq_len: int,
    loss_fn: Callable,
    base_precision: Any,
    increased_precision: Any,
    batch: dict,
    key: jax.random.PRNGKey,
):
    """
    Helper function to create a model and convert parameters if needed.
    """
    assert cfg.model.training_mode in ["normal", "forward", "forward_forward"]

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
    elif cfg.model.cell in ["eunn"]:
        cell_type = partial(
            EUNNCell,
            input_dim=batch["input"].shape[-1],
            norm_before_readout=cfg.model.norm_before_readout,
            freeze_recurrence=cfg.model.freeze_recurrence,
            dtype=dtype,
            n_layers=getattr(cfg.model, "eunn_n_layers", 4),
            nonlinearity=getattr(cfg.model, "eunn_nonlinearity", "none"),
        )
    else:
        raise ValueError(f"Unknown cell type: {cfg.model.cell}")

    BatchedRNN = nn.vmap(
        partial(
            StandardRNN,
            cell_type=cell_type,
            pooling=cfg.model.pooling,
            dtype=base_precision,
            unroll=cfg.model.unroll,
        ),
        in_axes=0,
        out_axes=0,
        variable_axes={"params": None},
        split_rngs={"params": False},
    )
    batched_model = BatchedRNN(
        hidden_dim=cfg.model.hidden_dim,
        output_dim=output_dim,
        dtype=base_precision,
    )
    params = batched_model.init(key, batch)["params"]

    if cfg.model.training_mode in ["forward", "forward_forward"]:
        # Overwrite the model to use the correct one, and convert parameters
        model = partial(
            ForwardBPTTRNN,
            cell=partial(
                ForwardBPTTCell,
                cell_type=cell_type,
                loss_fn=loss_fn,
                base_precision=base_precision,
                increased_precision=increased_precision,
                approx_inverse=cfg.model.approx_inverse,
                norm_before_readout=cfg.model.norm_before_readout,
                pooling=cfg.model.pooling,
            ),
            base_precision=base_precision,
            increased_precision=increased_precision,
            two_passes=cfg.model.training_mode == "forward_forward",
            pooling=cfg.model.pooling,
        )
        BatchedRNN = nn.vmap(
            model,
            in_axes=0,
            out_axes=0,
            variable_axes={"params": None},
            split_rngs={"params": False},
        )
        batched_model = BatchedRNN(
            hidden_dim=cfg.model.hidden_dim,
            output_dim=output_dim,
            base_precision=base_precision,
            increased_precision=increased_precision,
        )
        params = conversion_params_normal_to_forwardbptt(params, cell_name=cfg.model.cell.upper())

    return params, batched_model


def conversion_params_normal_to_forwardbptt(params: dict, cell_name: str = "GRU") -> dict:
    """
    Convert parameters from StandardRNN to ForwardBPTTRNN format.
    """
    return {"ScanForwardBPTTCell_0": {f"cell": params[f"rnn"]}}
