# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.utils.checkpoint import CheckpointPolicy

from torchtitan.experiments.graph_trainer.common_utils import (
    _get_module_fqn,
    _is_backward_node,
    annotate_module_fqns,
)
from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    minimal_fx_tracer,
    run_traced,
)
from torchtitan.experiments.graph_trainer.memory_policy import (
    _make_annotated_memory_policy,
    tag_sac_policy,
)
from torchtitan.experiments.graph_trainer.selective_activation_remat import (
    selective_activation_remat_pass,
)
from torchtitan.experiments.graph_trainer.tensor_ac import (
    _CHECKPOINT_OP,
    strip_tensor_checkpoint,
    TENSOR_AC_ANNOTATED_KEY,
    tensor_checkpoint,
)

DIM = 4
BATCH = 3
HEAD_DIM = 2
NUM_LAYERS = 2


class _Block(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        # Single-output norm, like the RMSNorm used in the llama3 blocks: the
        # annotation then names the norm op itself rather than a tuple slice.
        self.norm = nn.RMSNorm(dim)
        self.linear = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = tensor_checkpoint(self.norm(x))
        return x + F.silu(self.linear(h))


class _TupleNormModel(nn.Module):
    """LayerNorm returns a tuple, so the annotated tensor is a ``getitem``."""

    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(DIM)
        self.head = nn.Linear(DIM, HEAD_DIM, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = tensor_checkpoint(self.norm(x))
        return self.head(F.silu(h))


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Block(DIM) for _ in range(NUM_LAYERS)])
        self.head = nn.Linear(DIM, HEAD_DIM, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


def _step(model: nn.Module, x: torch.Tensor, target: torch.Tensor) -> list:
    prediction = model(x)
    loss = F.mse_loss(prediction, target)
    params = [p for p in model.parameters() if p.requires_grad]
    return [loss, *torch.autograd.grad(loss, params)]


def _trace(model: nn.Module) -> tuple:
    torch.manual_seed(0)
    x = torch.randn(BATCH, DIM)
    target = torch.randn(BATCH, HEAD_DIM)
    traced = minimal_fx_tracer(lambda inp, tgt: _step(model, inp, tgt), module=model)(
        x, target
    )
    return traced, x, target


class TestTensorAC(TestCase):
    def test_eager_identity_and_gradient(self) -> None:
        x = torch.randn(BATCH, DIM, requires_grad=True)
        y = tensor_checkpoint(x)

        self.assertEqual(y, x)
        self.assertEqual(y.untyped_storage().data_ptr(), x.untyped_storage().data_ptr())

        (2 * y).sum().backward()
        self.assertEqual(x.grad, torch.full_like(x, 2.0))

    def test_trace_strips_markers_and_tags_producers(self) -> None:
        model = _ToyModel()
        annotate_module_fqns(model)
        traced, x, target = _trace(model)
        graph = traced.gm.graph

        marker_nodes = [
            node for node in graph.nodes if node.target is _CHECKPOINT_OP
        ]
        self.assertEqual(len(marker_nodes), NUM_LAYERS)

        annotated = strip_tensor_checkpoint(traced.gm)

        self.assertEqual(len(annotated), NUM_LAYERS)
        self.assertEqual(
            [node for node in graph.nodes if node.target is _CHECKPOINT_OP], []
        )
        self.assertEqual(
            {_get_module_fqn(node) for node in annotated},
            {f"layers.{i}.norm" for i in range(NUM_LAYERS)},
        )
        for node in annotated:
            self.assertTrue(node.meta[TENSOR_AC_ANNOTATED_KEY])

        actual = run_traced(traced, module=model)(x, target)
        for actual_tensor, expected_tensor in zip(
            actual, _step(model, x, target), strict=True
        ):
            self.assertEqual(actual_tensor, expected_tensor)

    def test_annotated_policy_recomputes_only_marked_tensors(self) -> None:
        model = _ToyModel()
        annotate_module_fqns(model)
        traced, _, _ = _trace(model)
        annotated = strip_tensor_checkpoint(traced.gm)

        tag_sac_policy(traced.gm, policy_fn=_make_annotated_memory_policy())

        recompute = {
            node
            for node in traced.gm.graph.nodes
            if node.meta.get("recompute") == CheckpointPolicy.MUST_RECOMPUTE
        }
        saved = {
            node
            for node in traced.gm.graph.nodes
            if node.meta.get("recompute") == CheckpointPolicy.MUST_SAVE
        }
        self.assertLessEqual(annotated, recompute)
        self.assertTrue(saved)
        self.assertFalse(recompute & saved)
        self.assertFalse(
            [
                node
                for node in traced.gm.graph.nodes
                if _is_backward_node(node) and "recompute" in node.meta
            ]
        )

    def test_annotation_wins_over_getitem_parent_tagging(self) -> None:
        model = _TupleNormModel()
        annotate_module_fqns(model)
        traced, _, _ = _trace(model)
        annotated = strip_tensor_checkpoint(traced.gm)

        self.assertEqual(len(annotated), 1)
        (node,) = annotated
        self.assertEqual(node.target, operator.getitem)

        tag_sac_policy(traced.gm, policy_fn=_make_annotated_memory_policy())

        self.assertEqual(node.meta["recompute"], CheckpointPolicy.MUST_RECOMPUTE)
        # Inheritance would have copied the parent norm op's tag; the
        # annotation must win instead.
        self.assertNotEqual(
            node.args[0].meta.get("recompute"), CheckpointPolicy.MUST_RECOMPUTE
        )

    def test_remat_inserts_recompute_and_matches_eager_numerics(self) -> None:
        model = _ToyModel()
        annotate_module_fqns(model)
        traced, x, target = _trace(model)
        annotated = strip_tensor_checkpoint(traced.gm)

        tag_sac_policy(traced.gm, policy_fn=_make_annotated_memory_policy())
        selective_activation_remat_pass(traced.gm)

        dups = [
            node
            for node in traced.gm.graph.nodes
            if node.name.endswith("_recomputed")
        ]
        self.assertTrue(dups)
        recomputed_originals = {
            node.name.removesuffix("_recomputed") for node in dups
        }
        for node in annotated:
            self.assertIn(node.name, recomputed_originals)
            # The forward storage is no longer what backward reads.
            self.assertFalse([user for user in node.users if _is_backward_node(user)])

        actual = run_traced(traced, module=model)(x, target)
        for actual_tensor, expected_tensor in zip(
            actual, _step(model, x, target), strict=True
        ):
            self.assertEqual(actual_tensor, expected_tensor)


if __name__ == "__main__":
    run_tests()
