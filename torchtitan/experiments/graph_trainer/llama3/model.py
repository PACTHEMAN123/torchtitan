# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import types
from dataclasses import dataclass

import torch
from torch import nn

from torchtitan.experiments.graph_trainer.tensor_ac import tensor_checkpoint
from torchtitan.models.llama3 import Llama3Model

from ..simple_fsdp import disable_active_parametrization


class GraphTrainerLlama3Model(Llama3Model):
    @dataclass(kw_only=True, slots=True)
    class Config(Llama3Model.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)

    def init_states(
        self,
        *,
        buffer_device: torch.device | None = None,
    ) -> None:
        with disable_active_parametrization():
            super().init_states(buffer_device=buffer_device)


def _forward_with_tensor_ac(
    self,
    x: torch.Tensor,
    attention_masks,
    positions: torch.Tensor | None = None,
    *,
    padding_mask: torch.Tensor | None = None,
):
    del padding_mask
    h = x + self.attention(
        tensor_checkpoint(self.attention_norm(x)),
        attention_masks,
        positions,
    )
    out = h + self.feed_forward(tensor_checkpoint(self.ffn_norm(h)))
    return out


class _CheckpointedActivation(nn.Module):
    """SwiGLU activation whose output is annotated for tensor-level AC.

    The activation output is the widest tensor in the block (``[tokens,
    ffn_dim]``) and its only consumer is the ``w2`` projection, so dropping it
    after the forward pass and rebuilding it in backward trades a cheap
    elementwise recompute for the largest saved activation.
    """

    def __init__(self, activation_fn: nn.Module) -> None:
        super().__init__()
        self.activation_fn = activation_fn

    def forward(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        return tensor_checkpoint(self.activation_fn(gate, up))


def annotate_tensor_ac(model: GraphTrainerLlama3Model) -> int:
    """Mark per-block tensors for tensor-level activation checkpointing.

    Each block annotates its norm outputs (cheap to rebuild, single forward
    consumer) and the FFN activation output. Returns the number of annotated
    blocks.
    """
    layers = model.get_submodule("layers")
    count = 0
    for layer in layers.values():
        layer.forward = types.MethodType(_forward_with_tensor_ac, layer)
        feed_forward = layer.feed_forward
        feed_forward.activation_fn = _CheckpointedActivation(
            feed_forward.activation_fn
        )
        count += 1
    return count
