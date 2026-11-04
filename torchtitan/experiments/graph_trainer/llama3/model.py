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


def _annotate_module_output(module: nn.Module) -> None:
    """Wrap ``module.forward`` so its output is annotated in the traced graph.

    Wrapping the instance method instead of inserting a child module keeps the
    module's FQN (and therefore its layer id and boundary metadata) unchanged.
    """
    forward = module.forward

    def forward_with_annotation(self, *args, **kwargs):
        return tensor_checkpoint(forward(*args, **kwargs))

    module.forward = types.MethodType(forward_with_annotation, module)


def annotate_tensor_ac(model: GraphTrainerLlama3Model) -> int:
    """Mark per-block tensors for tensor-level activation checkpointing.

    Each block annotates its norm outputs (cheap to rebuild, single forward
    consumer) and the fused QKV projection output. The QKV output is the tensor
    the default SAC policy deliberately saves — attention backward reads q/k/v
    — so annotating it is what trades recompute for memory: the projection is
    rerun in backward instead of keeping ``[tokens, 3 * dim]`` alive per layer.
    Returns the number of annotated blocks.
    """
    layers = model.get_submodule("layers")
    count = 0
    for layer in layers.values():
        layer.forward = types.MethodType(_forward_with_tensor_ac, layer)
        # The split into q/k/v views happens above ``wqkv``; annotating the
        # linear output itself lets the recompute drop the whole tensor.
        _annotate_module_output(layer.attention.qkv_linear.wqkv)
        count += 1
    return count
