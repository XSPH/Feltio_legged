"""Utilities for exporting trained policies to deployment formats."""

import copy
import os
from pathlib import Path

import torch


def export_policy_as_jit(actor_critic, path):
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, 'policy_1.pt')
    model = StudentPolicy(actor_critic).to('cpu').eval()
    traced_script_module = torch.jit.script(model)
    traced_script_module.save(path)


class StudentPolicy(torch.nn.Module):
    """Deployment policy composed of the student encoder and shared actor."""

    def __init__(self, policy):
        super().__init__()
        self.student_encoder = copy.deepcopy(policy.estimator.source_encoder)
        self.actor = copy.deepcopy(policy.actor)
        self.num_actor_obs = policy.num_actor_obs

    def forward(self, history):
        current_obs = history[:, -self.num_actor_obs:]
        student_output = self.student_encoder(history)
        predicted_velocity = student_output[:, :3]
        latent = torch.nn.functional.normalize(
            student_output[:, 3:], p=2.0, dim=-1
        )
        context = torch.cat([predicted_velocity, latent], dim=-1)
        actor_input = torch.cat([current_obs, context], dim=-1)
        return self.actor(actor_input)


def export_policy_as_onnx(policy, path, filename="policy.onnx", verbose=False):
    """Export the history-based student policy to ONNX."""
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    model = StudentPolicy(policy).cpu().eval()
    history_dim = policy.history_length * policy.num_actor_obs
    history = torch.zeros(1, history_dim)
    torch.onnx.export(
        model,
        history,
        str(output_path),
        export_params=True,
        opset_version=11,
        verbose=verbose,
        input_names=["history"],
        output_names=["actions"],
        dynamic_axes={"history": {0: "batch"}, "actions": {0: "batch"}},
    )
    return output_path
