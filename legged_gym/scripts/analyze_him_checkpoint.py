"""Analyze HIM prototype and latent geometry without starting Isaac Gym."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rsl_rl.modules.estimator import (
    compute_prototype_metrics,
    compute_vector_geometry_metrics,
)


def get_args():
    parser = argparse.ArgumentParser(
        description="Analyze HIM representation geometry from a checkpoint."
    )
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument(
        "--latent_data_path",
        help="Optional .npz file containing a two-dimensional 'latents' array.",
    )
    parser.add_argument(
        "--output",
        help="Optional path for the JSON report.",
    )
    return parser.parse_args()


def load_checkpoint(checkpoint_path):
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must contain a state dictionary")
    return checkpoint


def get_model_state_dict(checkpoint):
    model_state_dict = checkpoint.get("model_state_dict")
    if model_state_dict is None:
        model_state_dict = checkpoint
    if not isinstance(model_state_dict, dict):
        raise ValueError("Checkpoint model_state_dict must be a dictionary")
    return model_state_dict


def find_prototypes(model_state_dict):
    matches = [
        (key, value)
        for key, value in model_state_dict.items()
        if key.endswith("estimator.prototypes.weight")
        or key.endswith("prototypes.weight")
    ]
    if len(matches) != 1:
        matched_keys = [key for key, _ in matches]
        raise KeyError(
            "Expected exactly one HIM prototype tensor, "
            f"found {len(matches)}: {matched_keys}"
        )
    key, prototypes = matches[0]
    if not isinstance(prototypes, torch.Tensor) or prototypes.ndim != 2:
        raise ValueError(f"Prototype value at '{key}' must be a 2D tensor")
    return key, prototypes.detach().cpu().float()


def metrics_to_float(metrics):
    return {key: float(value.item()) for key, value in metrics.items()}


def load_latents(latent_data_path, expected_dim):
    with np.load(latent_data_path, allow_pickle=False) as latent_data:
        if "latents" not in latent_data:
            raise KeyError("Latent data must contain a 'latents' array")
        latents = np.asarray(latent_data["latents"])
    if latents.ndim != 2:
        raise ValueError(
            f"Latents must have shape [samples, dimensions], got {latents.shape}"
        )
    if latents.shape[1] != expected_dim:
        raise ValueError(
            f"Latent dimension {latents.shape[1]} does not match "
            f"prototype dimension {expected_dim}"
        )
    return torch.from_numpy(latents).float()


def analyze_checkpoint(checkpoint_path, latent_data_path=None):
    checkpoint = load_checkpoint(checkpoint_path)
    model_state_dict = get_model_state_dict(checkpoint)
    prototype_key, prototypes = find_prototypes(model_state_dict)
    normalized_prototypes = torch.nn.functional.normalize(
        prototypes, p=2.0, dim=-1
    )

    iteration = checkpoint.get("iter")
    if isinstance(iteration, torch.Tensor):
        iteration = iteration.item()

    report = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_iteration": iteration,
        "prototype_state_key": prototype_key,
        "prototype_shape": list(prototypes.shape),
        "prototype_metrics": metrics_to_float(
            compute_prototype_metrics(prototypes)
        ),
        "prototype_cosine_matrix": (
            normalized_prototypes @ normalized_prototypes.T
        ).tolist(),
    }

    if latent_data_path is not None:
        latents = load_latents(latent_data_path, prototypes.shape[1])
        report["latent_data"] = str(latent_data_path)
        report["latent_shape"] = list(latents.shape)
        report["latent_metrics"] = metrics_to_float(
            compute_vector_geometry_metrics(latents)
        )
    return report


def main():
    args = get_args()
    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    latent_data_path = None
    if args.latent_data_path is not None:
        latent_data_path = Path(args.latent_data_path).expanduser().resolve()
        if not latent_data_path.is_file():
            raise FileNotFoundError(f"Latent data not found: {latent_data_path}")

    report = analyze_checkpoint(checkpoint_path, latent_data_path)
    report_json = json.dumps(report, indent=2, ensure_ascii=False)
    print(report_json)

    if args.output is not None:
        output_path = Path(args.output).expanduser().resolve()
        if output_path.suffix.lower() != ".json":
            output_path = output_path.with_suffix(".json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report_json + "\n", encoding="utf-8")
        print(f"Saved report to: {output_path}")


if __name__ == "__main__":
    main()
