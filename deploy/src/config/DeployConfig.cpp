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

template <std::size_t N>
std::array<float, N> requiredArray(const YAML::Node& node, const char* key) {
    const auto values = required<std::vector<float>>(node, key);
    if (values.size() != N) {
        throw std::runtime_error(std::string(key) + " must contain "
                                 + std::to_string(N) + " values");
    }
    std::array<float, N> result{};
    std::copy(values.begin(), values.end(), result.begin());
    return result;
}

LegGains requiredGains(const YAML::Node& node, const char* key) {
    const auto gains = required<YAML::Node>(node, key);
    return {requiredArray<3>(gains, "Kp"), requiredArray<3>(gains, "Kd")};
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
    const auto fel = required<YAML::Node>(root, "fel");
    cfg.fel.model_path = resolvePath(base, required<std::string>(fel, "model_path"));
    cfg.fel.model_input_size = required<int>(fel, "model_input_size");
    cfg.fel.model_output_size = required<int>(fel, "model_output_size");
    cfg.fel.model_num_threads = required<int>(fel, "model_num_threads");
    cfg.fel.default_dof_pos = requiredArray<12>(fel, "default_dof_pos");
    cfg.fel.scale_ang_vel = required<float>(fel, "scale_ang_vel");
    cfg.fel.scale_lin_vel_x = required<float>(fel, "scale_lin_vel_x");
    cfg.fel.scale_lin_vel_y = required<float>(fel, "scale_lin_vel_y");
    cfg.fel.scale_dof_pos = required<float>(fel, "scale_dof_pos");
    cfg.fel.scale_dof_vel = required<float>(fel, "scale_dof_vel");
    cfg.fel.observation_clamp = required<float>(fel, "observation_clamp");
    cfg.fel.action_clamp = required<float>(fel, "action_clamp");
    cfg.fel.actions_scale = required<float>(fel, "actions_scale");
    cfg.fel.hip_reduction = required<float>(fel, "hip_reduction");
    cfg.fel.stand_duration = required<float>(fel, "stand_duration");
    cfg.fel.speed_ramp_step = required<float>(fel, "speed_ramp_step");
    cfg.fel.num_proprio = required<int>(fel, "num_proprio");
    cfg.fel.history_len = required<int>(fel, "history_len");
    cfg.fel.num_actions = required<int>(fel, "num_actions");
    cfg.fel.sim_rl_gain = requiredGains(fel, "sim_rl_gain");
    cfg.fel.sim_stance_gain = requiredGains(fel, "sim_stance_gain");

    const auto simulation = required<YAML::Node>(root, "simulation");
    cfg.scene_path = resolvePath(base, required<std::string>(simulation, "scene"));
    cfg.simulation_timestep = required<double>(simulation, "timestep");
    cfg.simulation_decimation = required<int>(simulation, "decimation");
    cfg.render_hz = required<double>(simulation, "render_hz");
    cfg.initial_base_height = required<double>(simulation, "initial_base_height");

    const auto commands = required<YAML::Node>(root, "commands");
    cfg.max_linear_x = required<float>(commands, "max_linear_x");
    cfg.max_linear_y = required<float>(commands, "max_linear_y");
    cfg.max_angular_z = required<float>(commands, "max_angular_z");
    cfg.joystick_deadzone = required<float>(commands, "deadzone");
    cfg.joystick_device = required<std::string>(
        required<YAML::Node>(root, "joystick"), "device");

    if (cfg.fel.num_proprio != 45 || cfg.fel.history_len != 5 ||
        cfg.fel.num_actions != 12) {
        throw std::runtime_error(
            "Only 45 proprioceptions, 5 history frames and 12 actions are supported");
    }
    if (cfg.fel.model_input_size != cfg.fel.num_proprio * cfg.fel.history_len ||
        cfg.fel.model_output_size != cfg.fel.num_actions) {
        throw std::runtime_error("Only a 225-input, 12-output student policy is supported");
    }
    if (cfg.fel.model_num_threads < 1 || cfg.simulation_decimation < 1) {
        throw std::runtime_error("fel.model_num_threads and simulation.decimation must be >= 1");
    }
    requirePositive(cfg.simulation_timestep, "simulation.timestep");
    requirePositive(cfg.render_hz, "simulation.render_hz");
    requirePositive(cfg.fel.hip_reduction, "fel.hip_reduction");
    requirePositive(cfg.fel.stand_duration, "fel.stand_duration");
    requirePositive(cfg.fel.observation_clamp, "fel.observation_clamp");
    requirePositive(cfg.fel.action_clamp, "fel.action_clamp");
    requirePositive(cfg.fel.speed_ramp_step, "fel.speed_ramp_step");
    if (cfg.joystick_deadzone < 0.0F || cfg.joystick_deadzone >= 1.0F) {
        throw std::runtime_error("commands.deadzone must be in [0, 1)");
    }
    if (!std::filesystem::is_regular_file(cfg.scene_path)) {
        throw std::runtime_error("MuJoCo scene not found: " + cfg.scene_path.string());
    }
    if (!std::filesystem::is_regular_file(cfg.fel.model_path)) {
        throw std::runtime_error("MNN policy not found: " + cfg.fel.model_path.string());
    }
    return cfg;
}
