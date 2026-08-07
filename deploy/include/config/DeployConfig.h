#ifndef DEPLOY_CONFIG_H
#define DEPLOY_CONFIG_H

#include <array>
#include <filesystem>
#include <string>

struct DeployConfig {
    std::filesystem::path config_path;
    std::filesystem::path model_path;
    std::filesystem::path scene_path;

    int model_input_size = 45;
    int model_output_size = 12;
    int model_num_threads = 2;

    double simulation_timestep = 0.005;
    int simulation_decimation = 4;
    double render_hz = 60.0;
    double initial_base_height = 0.35;

    float action_scale = 0.25F;
    float hip_reduction = 1.0F;
    float stiffness = 20.0F;
    float damping = 0.5F;
    float stand_duration = 2.0F;
    float clip_observations = 100.0F;
    float clip_actions = 100.0F;
    std::array<float, 12> default_joint_angles{};

    float angular_velocity_scale = 0.25F;
    float joint_position_scale = 1.0F;
    float joint_velocity_scale = 0.05F;
    float command_linear_scale = 2.0F;
    float command_yaw_scale = 0.25F;

    float max_linear_x = 1.0F;
    float max_linear_y = 1.0F;
    float max_angular_z = 1.0F;
    float joystick_deadzone = 0.08F;
    std::string joystick_device = "/dev/input/js0";

    static DeployConfig loadDefault();
    static DeployConfig load(const std::filesystem::path& path);
};

#endif
