import torch
import torch.nn as nn
import torch.nn.functional as F


class Estimator(nn.Module):
    def __init__(
        self,
        temporal_steps,
        num_one_step_obs,
        response_dim,
        latent_dim=16,
        enc_hidden_dims=(512, 256, 128),
        tar_hidden_dims=(128, 64),
        activation="elu",
        num_prototypes=16,
        temperature=3.0,
        sinkhorn_epsilon=0.05,
        sinkhorn_iterations=3,
    ):
        super().__init__()
        self.temporal_steps = temporal_steps
        self.num_one_step_obs = num_one_step_obs
        self.response_dim = response_dim
        self.num_latent = latent_dim
        self.temperature = temperature
        self.sinkhorn_epsilon = sinkhorn_epsilon
        self.sinkhorn_iterations = sinkhorn_iterations

        source_input_dim = temporal_steps * num_one_step_obs
        source_layers = []
        for hidden_dim in enc_hidden_dims:
            source_layers.append(nn.Linear(source_input_dim, hidden_dim))
            source_layers.append(get_activation(activation))
            source_input_dim = hidden_dim
        source_layers.append(nn.Linear(source_input_dim, latent_dim + 3))
        self.source_encoder = nn.Sequential(*source_layers)

        target_input_dim = response_dim
        target_layers = []
        for hidden_dim in tar_hidden_dims:
            target_layers.append(nn.Linear(target_input_dim, hidden_dim))
            target_layers.append(get_activation(activation))
            target_input_dim = hidden_dim
        target_layers.append(nn.Linear(target_input_dim, latent_dim))
        self.target_encoder = nn.Sequential(*target_layers)

        self.prototypes = nn.Embedding(num_prototypes, latent_dim)

    def forward(self, history):
        predicted_velocity, latent = self.encode_history(history)
        return torch.cat([predicted_velocity, latent], dim=-1)

    def encode_history(self, history):
        flattened_history = history.flatten(start_dim=1)
        expected_dim = self.temporal_steps * self.num_one_step_obs
        if flattened_history.shape[-1] != expected_dim:
            raise ValueError(
                f"Expected history dimension {expected_dim}, "
                f"got {flattened_history.shape[-1]}"
            )
        encoded = self.source_encoder(flattened_history)
        predicted_velocity = encoded[..., :3]
        latent = F.normalize(encoded[..., 3:], p=2.0, dim=-1)
        return predicted_velocity, latent

    def encode_target(self, next_response):
        if next_response.shape[-1] != self.response_dim:
            raise ValueError(
                f"Expected response dimension {self.response_dim}, "
                f"got {next_response.shape[-1]}"
            )
        return F.normalize(self.target_encoder(next_response), p=2.0, dim=-1)

    def get_latent(self, history):
        predicted_velocity, latent = self.encode_history(history)
        return predicted_velocity.detach(), latent.detach()

    @torch.no_grad()
    def compute_diagnostics(
        self,
        history,
        next_response,
        valid_mask,
        groups=None,
        primary_slice=None,
    ):
        _, history_latent = self.encode_history(history)
        target_latent = self.encode_target(next_response)
        valid_mask = valid_mask.reshape(-1).bool()
        if primary_slice is None:
            primary_slice = (0, history_latent.shape[0])
        primary_start, primary_end = primary_slice
        normalized_prototypes = F.normalize(
            self.prototypes.weight, p=2.0, dim=-1
        )

        metrics = compute_prototype_metrics(normalized_prototypes)
        metrics.update(
            _compute_paired_latent_metrics(
                history_latent[primary_start:primary_end],
                target_latent[primary_start:primary_end],
                normalized_prototypes,
                valid_mask[primary_start:primary_end],
            )
        )
        if groups is not None:
            for group_name, (start, end) in groups.items():
                metrics.update(
                    _compute_paired_latent_metrics(
                        history_latent[start:end],
                        target_latent[start:end],
                        normalized_prototypes,
                        valid_mask[start:end],
                        prefix=f"{group_name}_",
                    )
                )
        return {key: value.detach() for key, value in metrics.items()}

    def compute_losses(
        self,
        history,
        velocity_target,
        next_response,
        valid_mask,
    ):
        predicted_velocity, history_latent = self.encode_history(history)
        target_latent = self.encode_target(next_response)
        velocity_loss = F.mse_loss(predicted_velocity, velocity_target)

        valid_mask = valid_mask.reshape(-1).bool()
        zero = velocity_loss.new_zeros(())
        swav_loss = zero
        prototype_perplexity = zero
        assignment_entropy = zero
        prototype_usage_min = zero
        prototype_usage_max = zero

        if torch.count_nonzero(valid_mask) > 1:
            with torch.no_grad():
                normalized_prototypes = F.normalize(
                    self.prototypes.weight, p=2.0, dim=-1
                )
                self.prototypes.weight.copy_(normalized_prototypes)

            history_scores = (
                history_latent[valid_mask] @ self.prototypes.weight.T
            )
            target_scores = target_latent[valid_mask] @ self.prototypes.weight.T

            with torch.no_grad():
                history_assignments = sinkhorn(
                    history_scores,
                    epsilon=self.sinkhorn_epsilon,
                    iterations=self.sinkhorn_iterations,
                )
                target_assignments = sinkhorn(
                    target_scores,
                    epsilon=self.sinkhorn_epsilon,
                    iterations=self.sinkhorn_iterations,
                )

            history_log_prob = F.log_softmax(
                history_scores / self.temperature, dim=-1
            )
            target_log_prob = F.log_softmax(
                target_scores / self.temperature, dim=-1
            )
            swav_loss = -0.5 * (
                history_assignments * target_log_prob
                + target_assignments * history_log_prob
            ).mean()

            with torch.no_grad():
                target_prob = target_log_prob.exp()
                prototype_usage = target_prob.mean(dim=0)
                marginal_entropy = -(
                    prototype_usage
                    * prototype_usage.clamp_min(1.0e-8).log()
                ).sum()
                prototype_perplexity = marginal_entropy.exp()
                assignment_entropy = -(
                    target_prob * target_prob.clamp_min(1.0e-8).log()
                ).sum(dim=-1).mean()
                prototype_usage_min = prototype_usage.min()
                prototype_usage_max = prototype_usage.max()

        with torch.no_grad():
            velocity_error = (predicted_velocity - velocity_target).abs().mean(0)
            latent_norm = history_latent.norm(p=2, dim=-1).mean()

        metrics = {
            "prototype_perplexity": prototype_perplexity.detach(),
            "prototype_assignment_entropy": assignment_entropy.detach(),
            "prototype_usage_min": prototype_usage_min.detach(),
            "prototype_usage_max": prototype_usage_max.detach(),
            "velocity_error_x": velocity_error[0].detach(),
            "velocity_error_y": velocity_error[1].detach(),
            "velocity_error_z": velocity_error[2].detach(),
            "latent_norm": latent_norm.detach(),
            "valid_swav_samples": valid_mask.sum().detach(),
        }
        return velocity_loss, swav_loss, metrics


@torch.no_grad()
def compute_prototype_metrics(prototypes):
    prototypes = F.normalize(prototypes.float(), p=2.0, dim=-1)
    effective_rank, stable_rank = _matrix_rank_metrics(prototypes)
    zero = prototypes.new_zeros(())

    if prototypes.shape[0] > 1:
        cosine = prototypes @ prototypes.T
        off_diagonal_mask = ~torch.eye(
            prototypes.shape[0], dtype=torch.bool, device=prototypes.device
        )
        off_diagonal_cosine = cosine[off_diagonal_mask]
        mean_abs_cosine = off_diagonal_cosine.abs().mean()
        same_direction_fraction = (
            off_diagonal_cosine > 0.99
        ).float().mean()
        collinear_fraction = (
            off_diagonal_cosine.abs() > 0.99
        ).float().mean()
    else:
        mean_abs_cosine = zero
        same_direction_fraction = zero
        collinear_fraction = zero

    return {
        "prototype_effective_rank": effective_rank,
        "prototype_stable_rank": stable_rank,
        "prototype_mean_abs_off_diagonal_cosine": mean_abs_cosine,
        "prototype_same_direction_fraction": same_direction_fraction,
        "prototype_collinear_fraction": collinear_fraction,
    }


@torch.no_grad()
def compute_vector_geometry_metrics(vectors, max_cosine_pairs=2048):
    vectors = vectors.float()
    zero = vectors.new_zeros(())
    if vectors.shape[0] == 0:
        return {
            "average_norm": zero,
            "mean_vector_norm": zero,
            "centered_effective_rank": zero,
            "random_pair_cosine_mean": zero,
        }

    average_norm = vectors.norm(p=2, dim=-1).mean()
    mean_vector_norm = vectors.mean(dim=0).norm(p=2)
    centered_effective_rank, _ = _matrix_rank_metrics(vectors, center=True)

    pair_count = min(vectors.shape[0] // 2, max_cosine_pairs)
    if pair_count > 0:
        normalized_vectors = F.normalize(vectors, p=2.0, dim=-1)
        generator = torch.Generator()
        generator.manual_seed(0)
        pair_indices = torch.randperm(
            vectors.shape[0], generator=generator
        )[:2 * pair_count].to(vectors.device)
        random_pair_cosine_mean = (
            normalized_vectors[pair_indices[:pair_count]]
            * normalized_vectors[pair_indices[pair_count:]]
        ).sum(dim=-1).mean()
    else:
        random_pair_cosine_mean = zero

    return {
        "average_norm": average_norm,
        "mean_vector_norm": mean_vector_norm,
        "centered_effective_rank": centered_effective_rank,
        "random_pair_cosine_mean": random_pair_cosine_mean,
    }


@torch.no_grad()
def compute_hard_assignment_metrics(scores):
    zero = scores.new_zeros(())
    if scores.shape[0] == 0:
        return {
            "hard_prototype_active_count": zero,
            "hard_prototype_perplexity": zero,
            "hard_prototype_usage_min": zero,
            "hard_prototype_usage_max": zero,
        }

    hard_indices = scores.argmax(dim=-1)
    usage = torch.bincount(
        hard_indices, minlength=scores.shape[-1]
    ).float()
    active_count = (usage > 0).sum().float()
    usage /= usage.sum()
    entropy = -(
        usage * usage.clamp_min(1.0e-12).log()
    ).sum()

    return {
        "hard_prototype_active_count": active_count,
        "hard_prototype_perplexity": entropy.exp(),
        "hard_prototype_usage_min": usage.min(),
        "hard_prototype_usage_max": usage.max(),
    }


def _matrix_rank_metrics(matrix, center=False):
    matrix = matrix.float()
    zero = matrix.new_zeros(())
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        return zero, zero
    if center:
        matrix = matrix - matrix.mean(dim=0, keepdim=True)

    eigenvalues = torch.linalg.eigvalsh(matrix.T @ matrix).clamp_min(0.0)
    total = eigenvalues.sum()
    probabilities = eigenvalues / total.clamp_min(1.0e-12)
    entropy = -(
        probabilities * probabilities.clamp_min(1.0e-12).log()
    ).sum()
    has_energy = total > torch.finfo(matrix.dtype).eps
    effective_rank = torch.where(has_energy, entropy.exp(), zero)
    stable_rank = torch.where(
        has_energy,
        total / eigenvalues.max().clamp_min(1.0e-12),
        zero,
    )
    return effective_rank, stable_rank


def _compute_paired_latent_metrics(
    history_latent,
    target_latent,
    prototypes,
    valid_mask,
    prefix="",
):
    valid_history_latent = history_latent[valid_mask]
    valid_target_latent = target_latent[valid_mask]
    history_metrics = compute_vector_geometry_metrics(valid_history_latent)
    target_metrics = compute_vector_geometry_metrics(valid_target_latent)
    hard_metrics = compute_hard_assignment_metrics(
        valid_target_latent @ prototypes.T
    )
    zero = history_latent.new_zeros(())
    if valid_history_latent.shape[0] > 0:
        positive_cosine = F.cosine_similarity(
            valid_history_latent, valid_target_latent, dim=-1
        ).mean()
    else:
        positive_cosine = zero

    metrics = {
        f"{prefix}source_latent_average_norm": history_metrics["average_norm"],
        f"{prefix}source_latent_mean_norm": history_metrics["mean_vector_norm"],
        f"{prefix}source_latent_centered_effective_rank": history_metrics[
            "centered_effective_rank"
        ],
        f"{prefix}source_latent_random_pair_cosine_mean": history_metrics[
            "random_pair_cosine_mean"
        ],
        f"{prefix}target_latent_average_norm": target_metrics["average_norm"],
        f"{prefix}target_latent_mean_norm": target_metrics["mean_vector_norm"],
        f"{prefix}target_latent_centered_effective_rank": target_metrics[
            "centered_effective_rank"
        ],
        f"{prefix}target_latent_random_pair_cosine_mean": target_metrics[
            "random_pair_cosine_mean"
        ],
        f"{prefix}positive_source_target_cosine": positive_cosine,
        f"{prefix}diagnostic_valid_samples": valid_mask.sum().float(),
    }
    for key, value in hard_metrics.items():
        metrics[f"{prefix}{key}"] = value
    return metrics


@torch.no_grad()
def sinkhorn(scores, epsilon=0.05, iterations=3):
    assignments = torch.exp(scores / epsilon).T
    num_prototypes, batch_size = assignments.shape
    assignments /= assignments.sum().clamp_min(1.0e-12)

    for _ in range(iterations):
        assignments /= assignments.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
        assignments /= num_prototypes
        assignments /= assignments.sum(dim=0, keepdim=True).clamp_min(1.0e-12)
        assignments /= batch_size

    return (assignments * batch_size).T


def get_activation(name):
    if name == "elu":
        return nn.ELU()
    if name == "selu":
        return nn.SELU()
    if name in ("relu", "crelu"):
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    if name == "lrelu":
        return nn.LeakyReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    raise ValueError(f"Invalid activation function: {name}")
