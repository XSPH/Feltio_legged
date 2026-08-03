#include "config/DeployConfig.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <yaml-cpp/yaml.h>

namespace {

template <typename T>
T required(const YAML::Node& node, const char* key) {
    if (!node || !node[key]) {
        throw std::runtime_error(std::string("Missing deploy config key: ") + key);
    }
    return node[key].as<T>();
}

std::filesystem::path resolvePath(const std::filesystem::path& base,
                                  const std::filesystem::path& value) {
    return value.is_absolute() ? value.lexically_normal()
                               : (base / value).lexically_normal();
}

void requirePositive(double value, const char* name) {
    if (!std::isfinite(value) || value <= 0.0) {
        throw std::runtime_error(std::string(name) + " must be positive");
    }
}

}  // namespace

DeployConfig DeployConfig::loadDefault() {
    std::vector<std::filesystem::path> candidates{
        std::filesystem::current_path() / "configs" / "config.yaml",
    };
    std::error_code error;
    const auto executable = std::filesystem::read_symlink("/proc/self/exe", error);
    if (!error) {
        candidates.push_back(executable.parent_path().parent_path()
                             / "configs" / "config.yaml");
        candidates.push_back(executable.parent_path()
                             / "configs" / "config.yaml");
    }
    for (const auto& candidate : candidates) {
        if (std::filesystem::is_regular_file(candidate)) {
            return load(candidate);
        }
    }
    throw std::runtime_error(
        "Default configs/config.yaml not found in the deploy directory");
}

DeployConfig DeployConfig::load(const std::filesystem::path& path) {
    DeployConfig cfg;
    cfg.config_path = std::filesystem::absolute(path).lexically_normal();

    YAML::Node root;
    try {
        root = YAML::LoadFile(cfg.config_path.string());
    } catch (const YAML::Exception& error) {
        throw std::runtime_error("Failed to load " + cfg.config_path.string() +
                                 ": " + error.what());
    }

    const auto base = cfg.config_path.parent_path();
    const auto model = required<YAML::Node>(root, "model");
    cfg.model_path = resolvePath(base, required<std::string>(model, "path"));
    cfg.model_input_size = required<int>(model, "input_size");
    cfg.model_output_size = required<int>(model, "output_size");
    cfg.model_num_threads = required<int>(model, "num_threads");

    const auto simulation = required<YAML::Node>(root, "simulation");
    cfg.scene_path = resolvePath(base, required<std::string>(simulation, "scene"));
    cfg.simulation_timestep = required<double>(simulation, "timestep");
    cfg.simulation_decimation = required<int>(simulation, "decimation");
    cfg.render_hz = required<double>(simulation, "render_hz");
    cfg.initial_base_height = required<double>(simulation, "initial_base_height");

    const auto controller = required<YAML::Node>(root, "controller");
    cfg.action_scale = required<float>(controller, "action_scale");
    cfg.stiffness = required<float>(controller, "stiffness");
    cfg.damping = required<float>(controller, "damping");
    cfg.stand_duration = required<float>(controller, "stand_duration");
    cfg.clip_observations = required<float>(controller, "clip_observations");
    cfg.clip_actions = required<float>(controller, "clip_actions");
    const auto defaults = required<std::vector<float>>(controller, "default_joint_angles");
    if (defaults.size() != cfg.default_joint_angles.size()) {
        throw std::runtime_error("controller.default_joint_angles must contain 12 values");
    }
    std::copy(defaults.begin(), defaults.end(), cfg.default_joint_angles.begin());

    const auto scales = required<YAML::Node>(root, "observation_scales");
    cfg.linear_velocity_scale = required<float>(scales, "linear_velocity");
    cfg.angular_velocity_scale = required<float>(scales, "angular_velocity");
    cfg.joint_position_scale = required<float>(scales, "joint_position");
    cfg.joint_velocity_scale = required<float>(scales, "joint_velocity");
    cfg.command_linear_scale = required<float>(scales, "command_linear");
    cfg.command_yaw_scale = required<float>(scales, "command_yaw");

    const auto commands = required<YAML::Node>(root, "commands");
    cfg.max_linear_x = required<float>(commands, "max_linear_x");
    cfg.max_linear_y = required<float>(commands, "max_linear_y");
    cfg.max_angular_z = required<float>(commands, "max_angular_z");
    cfg.joystick_deadzone = required<float>(commands, "deadzone");
    cfg.joystick_device = required<std::string>(
        required<YAML::Node>(root, "joystick"), "device");

    if (cfg.model_input_size != 48 || cfg.model_output_size != 12) {
        throw std::runtime_error("Only a 48-input, 12-output feed-forward actor is supported");
    }
    if (cfg.model_num_threads < 1 || cfg.simulation_decimation < 1) {
        throw std::runtime_error("model.num_threads and simulation.decimation must be >= 1");
    }
    requirePositive(cfg.simulation_timestep, "simulation.timestep");
    requirePositive(cfg.render_hz, "simulation.render_hz");
    requirePositive(cfg.stand_duration, "controller.stand_duration");
    requirePositive(cfg.clip_observations, "controller.clip_observations");
    requirePositive(cfg.clip_actions, "controller.clip_actions");
    if (cfg.joystick_deadzone < 0.0F || cfg.joystick_deadzone >= 1.0F) {
        throw std::runtime_error("commands.deadzone must be in [0, 1)");
    }
    if (!std::filesystem::is_regular_file(cfg.scene_path)) {
        throw std::runtime_error("MuJoCo scene not found: " + cfg.scene_path.string());
    }
    if (!std::filesystem::is_regular_file(cfg.model_path)) {
        throw std::runtime_error("MNN policy not found: " + cfg.model_path.string());
    }
    return cfg;
}
