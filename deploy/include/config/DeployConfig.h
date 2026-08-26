#ifndef DEPLOY_CONFIG_H
#define DEPLOY_CONFIG_H

#include <array>
#include <filesystem>
#include <string>

struct LegGains {
    std::array<float, 3> Kp{};
    std::array<float, 3> Kd{};
};

struct FelConfig {
    std::filesystem::path model_path;
    int model_input_size = 225;
    int model_output_size = 12;
    int model_num_threads = 2;

    std::array<float, 12> default_dof_pos{};
    float scale_ang_vel = 0.25F;
    float scale_lin_vel_x = 2.0F;
    float scale_lin_vel_y = 2.0F;
    float scale_dof_pos = 1.0F;
    float scale_dof_vel = 0.05F;

    float observation_clamp = 100.0F;
    float action_clamp = 100.0F;
    float actions_scale = 0.25F;
    float hip_reduction = 0.5F;
    float stand_duration = 2.0F;
    float speed_ramp_step = 0.05F;

    int num_proprio = 45;
    int history_len = 5;
    int num_actions = 12;

    LegGains sim_rl_gain;
    LegGains sim_stance_gain;
};

struct DeployConfig {
    std::filesystem::path config_path;
    std::filesystem::path scene_path;
    FelConfig fel;

    double simulation_timestep = 0.005;
    int simulation_decimation = 4;
    double render_hz = 60.0;
    double initial_base_height = 0.35;

    float max_linear_x = 1.0F;
    float max_linear_y = 1.0F;
    float max_angular_z = 1.0F;
    float joystick_deadzone = 0.08F;
    std::string joystick_device = "/dev/input/js0";

    static DeployConfig loadDefault();
    static DeployConfig load(const std::filesystem::path& path);
};

#endif
