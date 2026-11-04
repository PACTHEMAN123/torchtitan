# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tensor-level activation checkpointing annotations.

Model code writes the forward pass and marks individual tensors with
:func:`tensor_checkpoint`. The marker is an identity at runtime, and survives
``make_fx`` tracing as a ``tensor_ac.checkpoint`` node. The memory-policy pass
strips the marker and turns its producer into a ``MUST_RECOMPUTE`` boundary;
the existing selective-rematerialization pass then duplicates the minimal
producer subgraph in front of the backward consumer.

This follows the tensor-level AC described in the DeepSeek-V4 report: annotate
individual tensors instead of whole modules, and let the traced joint graph
decide what to recompute. Unlike the global SAC policies, unannotated
activations stay saved, so an annotation only trades recompute for the tensor
it names.
"""

import torch
import torch.fx as fx


TENSOR_AC_ANNOTATED_KEY = "tensor_ac_annotated"

# The library must outlive registration: recent torch versions deregister an
# op's schema when its ``Library`` object is collected.
_CHECKPOINT_LIB: torch.library.Library | None = None


def _register_checkpoint_op() -> "torch._ops.OpOverload":
    global _CHECKPOINT_LIB
    try:
        return torch.ops.tensor_ac.checkpoint.default
    except AttributeError:
        pass

    lib = _CHECKPOINT_LIB = torch.library.Library("tensor_ac", "FRAGMENT")
    lib.define("checkpoint(Tensor x) -> Tensor")
    # The marker returns its input without copying. ``torch.library.custom_op``
    # rejects aliasing returns, so register at the dispatcher level instead.
    lib.impl("checkpoint", lambda x: x, "CompositeExplicitAutograd")
    lib.impl("checkpoint", lambda x: x, "Meta")
    torch.library.register_autograd(
        "tensor_ac::checkpoint",
        lambda ctx, grad: grad,
        setup_context=lambda ctx, inputs, output: None,
    )
    return torch.ops.tensor_ac.checkpoint.default


_CHECKPOINT_OP = _register_checkpoint_op()


def tensor_checkpoint(x: torch.Tensor) -> torch.Tensor:
    """Annotate ``x`` for tensor-level activation checkpointing.

    Returns the same value. In a traced training graph the annotated tensor is
    released after its last forward use and recomputed before the backward
    consumer, together with any annotated producers it depends on. Without
    graph tracing (eager training, plain inference) this is an identity.
    """
    return _CHECKPOINT_OP(x)


def strip_tensor_checkpoint(gm: fx.GraphModule) -> set[fx.Node]:
    """Erase checkpoint markers and tag their producers.

    Returns the forward nodes that carried an annotation. Their metadata gets
    :data:`TENSOR_AC_ANNOTATED_KEY` so memory policies can read it after the
    marker nodes are gone.
    """
    annotated: set[fx.Node] = set()
    changed = False
    for node in list(gm.graph.nodes):
        if node.op != "call_function" or node.target is not _CHECKPOINT_OP:
            continue

        producer = node.args[0]
        if not isinstance(producer, fx.Node):
            raise TypeError(
                "tensor_checkpoint expects a traced tensor, got "
                f"{type(producer).__name__}"
            )
        producer.meta[TENSOR_AC_ANNOTATED_KEY] = True
        annotated.add(producer)
        node.replace_all_uses_with(producer)
        gm.graph.erase_node(node)
        changed = True

    if changed:
        gm.graph.lint()
        gm.recompile()
    return annotated
