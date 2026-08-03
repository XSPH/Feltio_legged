"""Utilities for exporting trained policies to deployment formats."""

import copy
from pathlib import Path

import torch


def _first_linear_input_dim(module):
    for layer in module.modules():
        if isinstance(layer, torch.nn.Linear):
            return layer.in_features
    raise ValueError("The policy actor does not contain a Linear input layer.")


def export_policy_as_onnx(policy, path, filename="policy.onnx", verbose=False):
    """Export a feed-forward actor-critic policy to ONNX."""
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    if not hasattr(policy, "actor"):
        raise ValueError("Policy does not have an actor module.")

    actor = copy.deepcopy(policy.actor).cpu().eval()
    input_dim = _first_linear_input_dim(actor)
    obs = torch.zeros(1, input_dim)
    torch.onnx.export(
        actor,
        obs,
        str(output_path),
        export_params=True,
        opset_version=11,
        verbose=verbose,
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
    )
    return output_path
