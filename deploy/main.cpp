#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

#include <GLFW/glfw3.h>
#include <mujoco/mujoco.h>

#include "config/DeployConfig.h"
#include "control/ControlFrame.h"
#include "control/CtrlComponents.h"
#include "interface/IOMujoco.h"
#include "telemetry/TelemetryLogger.h"

namespace {

mjModel* g_model = nullptr;
mjData* g_data = nullptr;
mjvCamera g_camera;
mjvOption g_option;
mjvScene g_scene;
mjvPerturb g_perturb;
mjrContext g_context;

bool g_left_button = false;
bool g_middle_button = false;
bool g_right_button = false;
double g_last_x = 0.0;
double g_last_y = 0.0;
double g_next_telemetry_print_time = 0.0;
std::atomic<bool> g_reset_requested{false};
std::atomic<bool> g_stop_requested{false};
std::atomic<UserCommand> g_keyboard_command{UserCommand::NONE};

constexpr std::array<const char*, 4> kFootGeomNames = {
    "FL_foot", "FR_foot", "RL_foot", "RR_foot",
};
std::array<int, kFootGeomNames.size()> g_foot_geom_ids{};

struct FootTelemetry {
    std::array<mjtNum, 3> position{};
    std::array<mjtNum, 3> velocity{};
    std::array<mjtNum, 3> contact_force{};
};
using FootTelemetryArray = std::array<FootTelemetry, kFootGeomNames.size()>;

struct KeyboardMovement {
    bool forward = false;
    bool backward = false;
    bool left = false;
    bool right = false;
    bool turn_left = false;
    bool turn_right = false;
};

KeyboardMovement g_keyboard_movement;

void requestStop(int) {
    g_stop_requested.store(true);
}

UserValue getKeyboardValue() {
    UserValue value;
    value.lx = static_cast<float>(g_keyboard_movement.right) -
               static_cast<float>(g_keyboard_movement.left);
    value.ly = static_cast<float>(g_keyboard_movement.backward) -
               static_cast<float>(g_keyboard_movement.forward);
    value.rx = static_cast<float>(g_keyboard_movement.turn_right) -
               static_cast<float>(g_keyboard_movement.turn_left);
    return value;
}

bool controlPressed(GLFWwindow* window) {
    return glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS ||
           glfwGetKey(window, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS;
}

void clearPerturbation() {
    g_perturb.select = 0;
    g_perturb.flexselect = -1;
    g_perturb.skinselect = -1;
    g_perturb.active = 0;
    g_perturb.active2 = 0;
    if (g_model != nullptr && g_data != nullptr) {
        mju_zero(g_data->xfrc_applied, 6 * g_model->nbody);
    }
}

void beginForceDrag(GLFWwindow* window, double x, double y) {
    int width = 0;
    int height = 0;
    glfwGetWindowSize(window, &width, &height);
    if (width <= 0 || height <= 0) {
        return;
    }

    mjtNum selection_point[3]{};
    int geom_id = -1;
    int flex_id = -1;
    int skin_id = -1;
    const int body_id = mjv_select(
        g_model, g_data, &g_option,
        static_cast<mjtNum>(width) / static_cast<mjtNum>(height),
        static_cast<mjtNum>(x) / static_cast<mjtNum>(width),
        1.0 - static_cast<mjtNum>(y) / static_cast<mjtNum>(height),
        &g_scene, selection_point, &geom_id, &flex_id, &skin_id);

    if (body_id <= 0) {
        clearPerturbation();
        return;
    }

    g_perturb.select = body_id;
    g_perturb.flexselect = flex_id;
    g_perturb.skinselect = skin_id;

    mjtNum offset[3];
    mju_sub3(offset, selection_point, g_data->xpos + 3 * body_id);
    mju_mulMatTVec(g_perturb.localpos, g_data->xmat + 9 * body_id, offset, 3, 3);
    mjv_initPerturb(g_model, g_data, &g_scene, &g_perturb);
    g_perturb.active = mjPERT_TRANSLATE;
}

void keyboard(GLFWwindow*, int key, int, int action, int) {
    const bool pressed = action != GLFW_RELEASE;
    if (key == GLFW_KEY_W) {
        g_keyboard_movement.forward = pressed;
    } else if (key == GLFW_KEY_S) {
        g_keyboard_movement.backward = pressed;
    } else if (key == GLFW_KEY_A) {
        g_keyboard_movement.left = pressed;
    } else if (key == GLFW_KEY_D) {
        g_keyboard_movement.right = pressed;
    } else if (key == GLFW_KEY_Q) {
        g_keyboard_movement.turn_left = pressed;
    } else if (key == GLFW_KEY_E) {
        g_keyboard_movement.turn_right = pressed;
    } else if (key == GLFW_KEY_SPACE && action == GLFW_PRESS) {
        g_keyboard_movement = KeyboardMovement{};
    }

    if (action != GLFW_PRESS) {
        return;
    }
    if (key == GLFW_KEY_BACKSPACE) {
        g_reset_requested.store(true);
    } else if (key == GLFW_KEY_P) {
        g_keyboard_command.store(UserCommand::PASS);
    } else if (key == GLFW_KEY_F) {
        g_keyboard_command.store(UserCommand::FIXED);
    } else if (key == GLFW_KEY_R) {
        g_keyboard_command.store(UserCommand::RL);
    }
}

void mouseButton(GLFWwindow* window, int button, int action, int) {
    g_left_button = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
    g_middle_button = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS;
    g_right_button = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
    glfwGetCursorPos(window, &g_last_x, &g_last_y);

    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS &&
        controlPressed(window)) {
        beginForceDrag(window, g_last_x, g_last_y);
    } else if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE) {
        clearPerturbation();
    }
}

void mouseMove(GLFWwindow* window, double x, double y) {
    if (!g_left_button && !g_middle_button && !g_right_button) {
        return;
    }
    const double dx = x - g_last_x;
    const double dy = y - g_last_y;
    g_last_x = x;
    g_last_y = y;

    int width = 0;
    int height = 0;
    glfwGetWindowSize(window, &width, &height);
    if (height <= 0) {
        return;
    }
    if (g_perturb.active != 0) {
        mjv_movePerturb(g_model, g_data, mjMOUSE_MOVE_V, dx / height,
                        dy / height, &g_scene, &g_perturb);
        return;
    }
    const bool shift = glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                       glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;
    mjtMouse action;
    if (g_right_button) {
        action = shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
    } else if (g_left_button) {
        action = shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
    } else {
        action = mjMOUSE_ZOOM;
    }
    mjv_moveCamera(g_model, action, dx / height, dy / height, &g_scene, &g_camera);
}

void scroll(GLFWwindow*, double, double y_offset) {
    mjv_moveCamera(g_model, mjMOUSE_ZOOM, 0.0, -0.05 * y_offset,
                   &g_scene, &g_camera);
}

int getRootJointId(const mjModel* model) {
    const int named_root = mj_name2id(model, mjOBJ_JOINT, "root");
    if (named_root >= 0) {
        if (model->jnt_type[named_root] != mjJNT_FREE) {
            throw std::runtime_error("MuJoCo joint 'root' must be a free joint");
        }
        return named_root;
    }
    for (int joint_id = 0; joint_id < model->njnt; ++joint_id) {
        if (model->jnt_type[joint_id] == mjJNT_FREE) {
            return joint_id;
        }
    }
    throw std::runtime_error("MuJoCo free root joint not found");
}

void initializePose(const mjModel* model, mjData* data, const DeployConfig& config) {
    mj_resetData(model, data);
    const int root_id = getRootJointId(model);
    const int root_qpos = model->jnt_qposadr[root_id];
    data->qpos[root_qpos + 0] = 0.0;
    data->qpos[root_qpos + 1] = 0.0;
    data->qpos[root_qpos + 2] = config.initial_base_height;
    data->qpos[root_qpos + 3] = 1.0;
    data->qpos[root_qpos + 4] = 0.0;
    data->qpos[root_qpos + 5] = 0.0;
    data->qpos[root_qpos + 6] = 0.0;

    constexpr const char* joint_names[12] = {
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    };
    for (int i = 0; i < 12; ++i) {
        const int joint_id = mj_name2id(model, mjOBJ_JOINT, joint_names[i]);
        if (joint_id < 0) {
            throw std::runtime_error(std::string("MuJoCo joint not found: ") +
                                     joint_names[i]);
        }
        data->qpos[model->jnt_qposadr[joint_id]] = config.fel.default_dof_pos[i];
    }
    mj_forward(model, data);
}

void initializeFootGeoms(const mjModel* model) {
    for (std::size_t i = 0; i < kFootGeomNames.size(); ++i) {
        g_foot_geom_ids[i] = mj_name2id(model, mjOBJ_GEOM, kFootGeomNames[i]);
        if (g_foot_geom_ids[i] < 0) {
            throw std::runtime_error(std::string("MuJoCo foot geom not found: ") +
                                     kFootGeomNames[i]);
        }
    }
}

FootTelemetryArray getFootTelemetry() {
    FootTelemetryArray telemetry{};
    for (std::size_t foot = 0; foot < telemetry.size(); ++foot) {
        const int geom_id = g_foot_geom_ids[foot];
        mju_copy3(telemetry[foot].position.data(), g_data->geom_xpos + 3 * geom_id);
        mjtNum spatial_velocity[6]{};
        mj_objectVelocity(g_model, g_data, mjOBJ_GEOM, geom_id, spatial_velocity, 0);
        mju_copy3(telemetry[foot].velocity.data(), spatial_velocity + 3);
    }

    for (int contact_id = 0; contact_id < g_data->ncon; ++contact_id) {
        const mjContact& contact = g_data->contact[contact_id];
        mjtNum contact_force[6]{};
        mj_contactForce(g_model, g_data, contact_id, contact_force);
        mjtNum world_force[3]{};
        mju_mulMatTVec(world_force, contact.frame, contact_force, 3, 3);
        for (std::size_t foot = 0; foot < telemetry.size(); ++foot) {
            if (contact.geom[0] == g_foot_geom_ids[foot]) {
                mju_subFrom3(telemetry[foot].contact_force.data(), world_force);
            }
            if (contact.geom[1] == g_foot_geom_ids[foot]) {
                mju_addTo3(telemetry[foot].contact_force.data(), world_force);
            }
        }
    }
    return telemetry;
}

void addVectorArrow(const std::array<mjtNum, 3>& origin,
                    const std::array<mjtNum, 3>& vector, mjtNum scale,
                    const float color[4]) {
    const mjtNum magnitude = mju_norm3(vector.data());
    if (magnitude < 1.0e-6 || g_scene.ngeom >= g_scene.maxgeom) {
        return;
    }
    constexpr mjtNum max_length = 0.35;
    const mjtNum arrow_scale = std::min(scale, max_length / magnitude);
    mjtNum endpoint[3];
    mju_scl3(endpoint, vector.data(), arrow_scale);
    mju_addTo3(endpoint, origin.data());

    mjvGeom& arrow = g_scene.geoms[g_scene.ngeom++];
    mjv_initGeom(&arrow, mjGEOM_ARROW, nullptr, nullptr, nullptr, color);
    mjv_connector(&arrow, mjGEOM_ARROW, 0.006, origin.data(), endpoint);
}

void addFootTelemetryGeoms(const FootTelemetryArray& telemetry) {
    constexpr mjtNum velocity_scale = 0.15;
    constexpr mjtNum force_scale = 0.002;
    constexpr float velocity_color[4] = {0.1F, 0.5F, 1.0F, 1.0F};
    constexpr float force_color[4] = {1.0F, 0.35F, 0.05F, 1.0F};
    for (const auto& value : telemetry) {
        addVectorArrow(value.position, value.velocity, velocity_scale, velocity_color);
        addVectorArrow(value.position, value.contact_force, force_scale, force_color);
    }
}

void drawFootTelemetryOverlay(mjrRect viewport, const FootTelemetryArray& telemetry) {
    std::ostringstream overlay;
    overlay << std::fixed << std::setprecision(2)
            << "Foot endpoint telemetry (world frame)\n"
            << "blue: velocity [m/s], orange: contact force [N]\n";
    for (std::size_t foot = 0; foot < telemetry.size(); ++foot) {
        const auto& value = telemetry[foot];
        overlay << kFootGeomNames[foot] << "  v "
                << value.velocity[0] << ' ' << value.velocity[1] << ' '
                << value.velocity[2] << "  |v| " << mju_norm3(value.velocity.data())
                << "\n    F " << value.contact_force[0] << ' '
                << value.contact_force[1] << ' ' << value.contact_force[2]
                << "  |F| " << mju_norm3(value.contact_force.data()) << '\n';
    }
    mjr_overlay(mjFONT_NORMAL, mjGRID_TOPLEFT, viewport, overlay.str().c_str(), "",
                &g_context);
}

void printFootTelemetry(const FootTelemetryArray& telemetry) {
    std::ostringstream output;
    output << std::fixed << std::setprecision(3) << "[feet] t=" << g_data->time;
    for (std::size_t foot = 0; foot < telemetry.size(); ++foot) {
        const auto& value = telemetry[foot];
        output << " | " << kFootGeomNames[foot] << " v=("
               << value.velocity[0] << ',' << value.velocity[1] << ','
               << value.velocity[2] << ") m/s F=(" << value.contact_force[0]
               << ',' << value.contact_force[1] << ',' << value.contact_force[2]
               << ") N";
    }
    std::cout << output.str() << std::endl;
}

void render(GLFWwindow* window) {
    mjrRect viewport{0, 0, 0, 0};
    glfwGetFramebufferSize(window, &viewport.width, &viewport.height);
    mjv_updateScene(g_model, g_data, &g_option, &g_perturb, &g_camera, mjCAT_ALL,
                    &g_scene);
    const auto telemetry = getFootTelemetry();
    if (g_data->time + 1.0e-9 >= g_next_telemetry_print_time) {
        printFootTelemetry(telemetry);
        g_next_telemetry_print_time = g_data->time + 0.1;
    }
    addFootTelemetryGeoms(telemetry);
    mjr_render(viewport, &g_scene, &g_context);
    drawFootTelemetryOverlay(viewport, telemetry);
    glfwSwapBuffers(window);
}

}  // namespace

int main() {
    GLFWwindow* window = nullptr;
    try {
        const DeployConfig config = DeployConfig::loadDefault();
        std::cout << "[config] " << config.config_path << '\n'
                  << "[scene]  " << config.scene_path << '\n'
                  << "[model]  " << config.fel.model_path << '\n';

        char error[1024]{};
        g_model = mj_loadXML(config.scene_path.c_str(), nullptr, error, sizeof(error));
        if (g_model == nullptr) {
            throw std::runtime_error(std::string("Failed to load MuJoCo scene: ") + error);
        }
        g_model->opt.timestep = config.simulation_timestep;
        g_data = mj_makeData(g_model);
        if (g_data == nullptr) {
            throw std::runtime_error("Failed to allocate MuJoCo data");
        }
        initializePose(g_model, g_data, config);
        initializeFootGeoms(g_model);

        if (!glfwInit()) {
            throw std::runtime_error("Failed to initialize GLFW");
        }
        window = glfwCreateWindow(1200, 900, "Legged Robot RL MuJoCo", nullptr, nullptr);
        if (window == nullptr) {
            throw std::runtime_error("Failed to create GLFW window");
        }
        glfwMakeContextCurrent(window);
        glfwSwapInterval(0);

        mjv_defaultCamera(&g_camera);
        mjv_defaultOption(&g_option);
        mjv_defaultScene(&g_scene);
        mjv_defaultPerturb(&g_perturb);
        mjr_defaultContext(&g_context);

        // Move ungrouped robot collision geoms to the hidden collision group.
        for (int geom_id = 0; geom_id < g_model->ngeom; ++geom_id) {
            if (g_model->geom_bodyid[geom_id] != 0 &&
                g_model->geom_group[geom_id] == 0) {
                g_model->geom_group[geom_id] = 3;
            }
        }
        g_option.geomgroup[3] = 0;
        g_option.flags[mjVIS_PERTFORCE] = 1;
        mjv_makeScene(g_model, &g_scene, 2000);
        mjr_makeContext(g_model, &g_context, mjFONTSCALE_150);
        g_camera.type = mjCAMERA_TRACKING;
        const int root_id = getRootJointId(g_model);
        g_camera.trackbodyid = g_model->jnt_bodyid[root_id];
        g_camera.distance = 2.0;
        g_camera.azimuth = 135.0;
        g_camera.elevation = -20.0;

        glfwSetKeyCallback(window, keyboard);
        glfwSetCursorPosCallback(window, mouseMove);
        glfwSetMouseButtonCallback(window, mouseButton);
        glfwSetScrollCallback(window, scroll);

        IOMujoco *ioInter = new IOMujoco(g_data, g_model, &config);
        CtrlComponents ctrlComp(ioInter, &config);
        ControlFrame ctrlFrame(&ctrlComp);
        TelemetryLogger telemetryLogger(g_model, g_data, config);
        std::signal(SIGINT, requestStop);

        std::cout << "Controls: B/F=fixed stand, A/R=RL, Y/P=passive, "
                     "W/S=forward/backward, A/D=left/right, "
                     "Q/E=turn left/right, Space=stop keyboard command, "
                     "Ctrl+left drag=apply force, Backspace=reset\n";
        const double policy_dt =
            config.simulation_timestep * config.simulation_decimation;
        const double render_dt = 1.0 / config.render_hz;
        double next_render_time = g_data->time;
        auto wall_deadline = std::chrono::steady_clock::now();

        while (!glfwWindowShouldClose(window) && !g_stop_requested.load()) {
            if (g_reset_requested.exchange(false)) {
                clearPerturbation();
                g_keyboard_movement = KeyboardMovement{};
                telemetryLogger.resetEpisode();
                initializePose(g_model, g_data, config);
                ctrlFrame.reset();
                next_render_time = g_data->time;
                g_next_telemetry_print_time = g_data->time;
                wall_deadline = std::chrono::steady_clock::now();
            }
            const UserCommand keyboard_command =
                g_keyboard_command.exchange(UserCommand::NONE);
            if (keyboard_command != UserCommand::NONE) {
                ctrlComp.setUserCommand(keyboard_command);
            }
            ioInter->setKeyboardValue(getKeyboardValue());

            ctrlFrame.run();
            mju_zero(g_data->xfrc_applied, 6 * g_model->nbody);
            if (g_perturb.active != 0) {
                mjv_applyPerturbForce(g_model, g_data, &g_perturb);
            }
            for (int i = 0; i < config.simulation_decimation; ++i) {
                if (i > 0) {
                    // Hold the policy target for one policy period, but refresh
                    // the PD torque from the latest state every physics step.
                    ctrlComp.sendRecv();
                    ctrlComp.send();
                }
                mj_step(g_model, g_data);
                telemetryLogger.sample(
                    ctrlComp, ctrlFrame._FSMController->_currentState->_stateName);
            }

            if (g_data->time + 1.0e-9 >= next_render_time) {
                render(window);
                glfwPollEvents();
                next_render_time += render_dt;
            }

            wall_deadline += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                std::chrono::duration<double>(policy_dt));
            std::this_thread::sleep_until(wall_deadline);
        }

        telemetryLogger.finalize();

        mjv_freeScene(&g_scene);
        mjr_freeContext(&g_context);
        glfwDestroyWindow(window);
        glfwTerminate();
        mj_deleteData(g_data);
        mj_deleteModel(g_model);
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "[fatal] " << error.what() << '\n';
        if (window != nullptr) {
            glfwDestroyWindow(window);
        }
        glfwTerminate();
        if (g_data != nullptr) {
            mj_deleteData(g_data);
        }
        if (g_model != nullptr) {
            mj_deleteModel(g_model);
        }
        return EXIT_FAILURE;
    }
}
