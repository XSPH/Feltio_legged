"""Visualize the learned HIM latent space across terrain types with t-SNE."""

import inspect
from pathlib import Path

import isaacgym  # noqa: F401  # Isaac Gym must be imported before torch.
from isaacgym import gymutil
import matplotlib
import numpy as np
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
import legged_gym.envs  # noqa: F401  # Register tasks.
from legged_gym.utils import task_registry
from legged_gym.utils.helpers import get_load_path


TERRAIN_NAMES = np.array(["Slope", "Rough slope", "Stairs", "Discrete"])
TERRAIN_COLORS = {
    0: "#55A868",
    1: "#DD8452",
    2: "#4C72B0",
    3: "#C44E52",
}
PLOT_ORDER = (2, 1, 0, 3)


def get_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "go2"},
        {"name": "--checkpoint_path", "type": str},
        {"name": "--load_run", "type": str},
        {"name": "--checkpoint", "type": int},
        {"name": "--num_envs", "type": int, "default": 2048},
        {"name": "--steps", "type": int, "default": 300},
        {"name": "--warmup_steps", "type": int, "default": 250},
        {"name": "--sample_interval", "type": int, "default": 4},
        {"name": "--samples_per_terrain", "type": int, "default": 500},
        {"name": "--command_x", "type": float, "default": 0.5},
        {"name": "--perplexity", "type": float, "default": 30.0},
        {"name": "--seed", "type": int, "default": 1},
        {
            "name": "--encoder",
            "type": str,
            "default": "student",
            "choices": ["student", "teacher"],
        },
        {"name": "--output", "type": str},
        {"name": "--show_sim", "action": "store_true", "default": False},
        {"name": "--rl_device", "type": str, "default": "cuda:0"},
    ]
    args = gymutil.parse_arguments(
        description="HIM latent-space visualization",
        custom_parameters=custom_parameters,
    )
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += f":{args.sim_device_id}"
    args.headless = not args.show_sim

    # Attributes expected by task_registry.make_alg_runner().
    args.resume = False
    args.experiment_name = None
    args.run_name = None
    args.max_iterations = None
    return args


def validate_args(args):
    if args.num_envs < 8:
        raise ValueError("--num_envs must be at least 8 to cover all terrain types")
    if args.steps <= args.warmup_steps:
        raise ValueError("--steps must be greater than --warmup_steps")
    if args.warmup_steps < 0:
        raise ValueError("--warmup_steps cannot be negative")
    if args.sample_interval <= 0:
        raise ValueError("--sample_interval must be positive")
    if args.samples_per_terrain <= 0:
        raise ValueError("--samples_per_terrain must be positive")
    if args.perplexity <= 0:
        raise ValueError("--perplexity must be positive")


def configure_environment(env_cfg, args):
    env_cfg.env.num_envs = args.num_envs
    env_cfg.env.test = True

    # Eight columns give two columns to each paper category. The two stair
    # columns contain ascending and descending stairs and share one label.
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 8
    env_cfg.terrain.max_init_terrain_level = 4
    env_cfg.terrain.curriculum = True
    env_cfg.terrain.terrain_proportions = [
        0.25,
        0.25,
        0.125,
        0.125,
        0.25,
    ]

    env_cfg.noise.add_noise = False
    env_cfg.commands.heading_command = False
    env_cfg.commands.resampling_time = 1.0e9

    domain_rand = env_cfg.domain_rand
    domain_rand.randomize_friction = False
    domain_rand.randomize_base_mass = False
    domain_rand.randomize_link_mass = False
    domain_rand.randomize_base_com = False
    domain_rand.randomize_restitution = False
    domain_rand.randomize_pd_gains = False
    domain_rand.randomize_motor_zero_offset = False
    domain_rand.randomize_motor_strength = False
    domain_rand.randomize_action_delay = False
    domain_rand.push_robots = False


def resolve_checkpoint(args, train_cfg):
    if args.checkpoint_path is not None:
        checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
    else:
        log_root = Path(LEGGED_GYM_ROOT_DIR) / "logs" / train_cfg.runner.experiment_name
        load_run = train_cfg.runner.load_run if args.load_run is None else args.load_run
        checkpoint = (
            train_cfg.runner.checkpoint
            if args.checkpoint is None
            else args.checkpoint
        )
        checkpoint_path = Path(
            get_load_path(str(log_root), load_run=load_run, checkpoint=checkpoint)
        ).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Policy checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def set_command(env, obs, privileged_obs, command_x):
    env.commands[:, :3] = 0.0
    env.commands[:, 0] = command_x
    scaled_commands = env.commands[:, :3] * env.commands_scale
    obs[:, 6:9] = scaled_commands
    if privileged_obs is not None:
        privileged_obs[:, 6:9] = scaled_commands


def get_terrain_labels(env):
    choices = env.terrain_types.detach().cpu().numpy() / env.cfg.terrain.num_cols
    choices += 0.001
    proportions = np.cumsum(env.cfg.terrain.terrain_proportions)
    labels = np.full(choices.shape, -1, dtype=np.int64)
    labels[choices < proportions[0]] = 0
    labels[(choices >= proportions[0]) & (choices < proportions[1])] = 1
    labels[(choices >= proportions[1]) & (choices < proportions[3])] = 2
    labels[(choices >= proportions[3]) & (choices < proportions[4])] = 3
    return labels


def collect_latents(env, actor_critic, history_length, args):
    obs = env.get_observations()
    privileged_obs = env.get_privileged_observations()
    history = torch.zeros(
        env.num_envs,
        history_length,
        env.num_obs,
        device=env.device,
    )
    history_age = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    terrain_labels = get_terrain_labels(env)
    collected = {label: [] for label in range(len(TERRAIN_NAMES))}
    sample_counts = np.zeros(len(TERRAIN_NAMES), dtype=np.int64)

    set_command(env, obs, privileged_obs, args.command_x)
    actor_critic.eval()
    with torch.inference_mode():
        for step in range(args.steps):
            history = torch.cat([history[:, 1:], obs.unsqueeze(1)], dim=1)
            history_age += 1
            student_context = actor_critic.encode_student(history)
            actor_input = torch.cat([obs, student_context], dim=-1)
            actions = actor_critic.actor(actor_input)

            if args.encoder == "student":
                latent = student_context[:, 3:]
            else:
                latent = actor_critic.encode_teacher(privileged_obs)[:, 3:]

            should_sample = (
                step >= args.warmup_steps
                and (step - args.warmup_steps) % args.sample_interval == 0
            )
            if should_sample:
                valid_history = history_age >= history_length
                latent_np = latent.detach().cpu().numpy()
                valid_history_np = valid_history.detach().cpu().numpy()
                for label in range(len(TERRAIN_NAMES)):
                    mask = (terrain_labels == label) & valid_history_np
                    if np.any(mask):
                        collected[label].append(latent_np[mask])
                        sample_counts[label] += np.count_nonzero(mask)
                if np.all(sample_counts >= args.samples_per_terrain):
                    break

            obs, privileged_obs, _, dones, _ = env.step(actions)
            reset_mask = dones > 0
            history[reset_mask] = 0.0
            history_age[reset_mask] = 0
            set_command(env, obs, privileged_obs, args.command_x)

    missing = [
        TERRAIN_NAMES[label]
        for label, samples in collected.items()
        if not samples
    ]
    if missing:
        raise RuntimeError("No latent samples collected for: " + ", ".join(missing))

    rng = np.random.default_rng(args.seed)
    balanced_latents = []
    balanced_labels = []
    for label, samples in collected.items():
        terrain_latents = np.concatenate(samples, axis=0)
        sample_count = min(args.samples_per_terrain, len(terrain_latents))
        indices = rng.choice(len(terrain_latents), sample_count, replace=False)
        balanced_latents.append(terrain_latents[indices])
        balanced_labels.append(np.full(sample_count, label, dtype=np.int64))

    latents = np.concatenate(balanced_latents, axis=0)
    labels = np.concatenate(balanced_labels, axis=0)
    permutation = rng.permutation(len(latents))
    return latents[permutation], labels[permutation]


def run_tsne(latents, perplexity, seed):
    try:
        from sklearn.manifold import TSNE
    except ImportError as error:
        raise RuntimeError(
            "scikit-learn is required for t-SNE; install it with "
            "`python -m pip install scikit-learn`"
        ) from error

    effective_perplexity = min(perplexity, float(len(latents) - 1))
    kwargs = {
        "n_components": 2,
        "perplexity": effective_perplexity,
        "learning_rate": "auto",
        "init": "pca",
        "random_state": seed,
    }
    if "max_iter" in inspect.signature(TSNE).parameters:
        kwargs["max_iter"] = 1000
    else:
        kwargs["n_iter"] = 1000
    print(
        f"Running t-SNE on {len(latents)} samples "
        f"with perplexity {effective_perplexity:g}..."
    )
    return TSNE(**kwargs).fit_transform(latents)


def plot_embedding(embedding, labels, output_path, encoder, show_plot):
    if not show_plot:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    for label in PLOT_ORDER:
        mask = labels == label
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=10,
            alpha=0.62,
            linewidths=0,
            color=TERRAIN_COLORS[label],
            label=TERRAIN_NAMES[label],
        )
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_title(f"HIM {encoder} latent space")
    ax.set_xticks([])
    ax.set_yticks([])
    legend = ax.legend(title="Terrain type", frameon=False, loc="best")
    legend_handles = getattr(legend, "legend_handles", None)
    if legend_handles is None:
        legend_handles = legend.legendHandles
    for handle in legend_handles:
        handle.set_sizes([42])
        handle.set_alpha(1.0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close(fig)


def get_output_path(args, checkpoint_path):
    if args.output is None:
        filename = f"{checkpoint_path.stem}_{args.encoder}_latent_tsne.png"
        return checkpoint_path.parent / filename
    output_path = Path(args.output).expanduser().resolve()
    if output_path.suffix.lower() != ".png":
        output_path = output_path.with_suffix(".png")
    return output_path


def destroy_environment(env):
    if env.viewer is not None:
        env.gym.destroy_viewer(env.viewer)
    env.gym.destroy_sim(env.sim)


def main():
    args = get_args()
    validate_args(args)
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    configure_environment(env_cfg, args)
    checkpoint_path = resolve_checkpoint(args, train_cfg)

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    train_cfg.runner.resume = False
    runner, train_cfg = task_registry.make_alg_runner(
        env=env,
        name=args.task,
        args=args,
        train_cfg=train_cfg,
        log_root=None,
    )
    print(f"Loading policy from: {checkpoint_path}")
    runner.load(str(checkpoint_path), load_optimizer=False)

    try:
        latents, labels = collect_latents(
            env,
            runner.alg.actor_critic,
            train_cfg.history_length,
            args,
        )
    finally:
        destroy_environment(env)

    output_path = get_output_path(args, checkpoint_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data_path = output_path.with_suffix(".npz")
    np.savez_compressed(
        data_path,
        latents=latents,
        labels=labels,
        terrain_names=TERRAIN_NAMES,
        checkpoint=str(checkpoint_path),
        encoder=args.encoder,
    )

    embedding = run_tsne(latents, args.perplexity, args.seed)
    np.savez_compressed(
        data_path,
        latents=latents,
        embedding=embedding,
        labels=labels,
        terrain_names=TERRAIN_NAMES,
        checkpoint=str(checkpoint_path),
        encoder=args.encoder,
    )
    plot_embedding(embedding, labels, output_path, args.encoder, args.show_sim)
    counts = {
        TERRAIN_NAMES[label]: int(np.count_nonzero(labels == label))
        for label in range(len(TERRAIN_NAMES))
    }
    print(f"Saved latent-space figure to: {output_path}")
    print(f"Saved latent samples and embedding to: {data_path}")
    print(f"Samples per terrain: {counts}")


if __name__ == "__main__":
    main()
