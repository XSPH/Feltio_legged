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
