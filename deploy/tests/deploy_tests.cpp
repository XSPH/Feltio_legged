#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

#include <mujoco/mujoco.h>

#include "FSM/FSM.h"
#include "FSM/State_Rl.h"
#include "config/DeployConfig.h"
#include "control/CtrlComponents.h"
#include "control/rl_Inference.h"
#include "interface/IOMujoco.h"

namespace{

class FakeIO final : public IOInterface{
public:
    FakeIO(){
        cmdPanel = std::make_unique<CmdPanel>();
    }

    void sendRecv(LowlevelCmd*, LowlevelState *state) override{
        state->userCmd = userCommand;
        state->userValue = userValue;
    }

    void send(LowlevelCmd*) override{}
    void recv(LowlevelState*) override{}

    UserCommand userCommand = UserCommand::PASS;
    UserValue userValue;
};

void require(bool condition, const std::string& message){
    if(!condition){
        throw std::runtime_error(message);
    }
}

void requireNear(float actual, float expected, const std::string& message){
    if(std::abs(actual - expected) > 1.0e-4F){
        throw std::runtime_error(message + ": expected " + std::to_string(expected)
                                 + ", got " + std::to_string(actual));
    }
}

float applyDeadzone(float value, float deadzone){
    if(std::abs(value) <= deadzone){
        return 0;
    }
    const float magnitude = (std::abs(value) - deadzone) / (1 - deadzone);
    return std::copysign(magnitude, value);
}

void buildObservation(const DeployConfig& config, const LowlevelState& lowState,
                      const UserValue& userValue,
                      const float lastAction[NUM_ACTIONS],
                      float observation[NUM_OBSERVATIONS]){
    const FelConfig& cfg = config.fel;
    for(int i = 0; i < 3; i++){
        observation[i] = lowState.imu.gyroscope[i] * cfg.scale_ang_vel;
    }
    observation[3] = 0;
    observation[4] = -1;
    observation[5] = 0;
    observation[6] = -userValue.ly * config.max_linear_x * cfg.scale_lin_vel_x;
    observation[7] = -userValue.lx * config.max_linear_y * cfg.scale_lin_vel_y;
    observation[8] = -userValue.rx * config.max_angular_z * cfg.scale_ang_vel;
    for(int i = 0; i < NUM_ACTIONS; i++){
        observation[i + 9] = (lowState.motorState[i].q
                              - cfg.default_dof_pos[i]) * cfg.scale_dof_pos;
        observation[i + 21] = lowState.motorState[i].dq * cfg.scale_dof_vel;
        observation[i + 33] = lastAction[i];
    }
}

void verifyMotorTargets(const FelConfig& cfg, const float action[NUM_ACTIONS],
                        const LowlevelCmd& lowCmd){
    for(int i = 0; i < NUM_ACTIONS; i++){
        const float clipped = std::clamp(action[i], -cfg.action_clamp,
                                         cfg.action_clamp);
        const float reduction = i % 3 == 0 ? cfg.hip_reduction : 1.0F;
        const float expected = cfg.default_dof_pos[i]
                             + clipped * reduction * cfg.actions_scale;
        requireNear(lowCmd.motorCmd[i].q, expected,
                    "RL motor target mismatch at index " + std::to_string(i));
        requireNear(lowCmd.motorCmd[i].dq, 0,
                    "RL motor velocity mismatch at index " + std::to_string(i));
    }
}

void testStateRlInterface(const DeployConfig& config){
    auto *fake = new FakeIO();
    CtrlComponents ctrlComp(fake, &config);
    State_Rl state(&ctrlComp);
    LowlevelState& lowState = *ctrlComp.lowState;
    const float halfSqrtTwo = std::sqrt(0.5F);
    lowState.imu.quaternion[0] = halfSqrtTwo;
    lowState.imu.quaternion[1] = halfSqrtTwo;
    lowState.imu.gyroscope[0] = 0.4F;
    lowState.imu.gyroscope[1] = -0.8F;
    lowState.imu.gyroscope[2] = 1.2F;
    for(int i = 0; i < NUM_ACTIONS; i++){
        lowState.motorState[i].q = config.fel.default_dof_pos[i] + 0.01F * i;
        lowState.motorState[i].dq = -0.1F * i;
    }
    lowState.userValue.lx = 0.1F;
    lowState.userValue.ly = -0.6F;
    lowState.userValue.rx = 0.7F;

    state.enter();
    state.run();

    UserValue rampedValue;
    rampedValue.lx = applyDeadzone(lowState.userValue.lx,
                                   config.joystick_deadzone);
    rampedValue.ly = -config.fel.speed_ramp_step;
    rampedValue.rx = config.fel.speed_ramp_step;
    float lastAction[NUM_ACTIONS]{};
    float observation[NUM_OBSERVATIONS]{};
    float history[NUM_POLICY_INPUTS]{};
    float expectedAction[NUM_ACTIONS]{};
    buildObservation(config, lowState, rampedValue, lastAction, observation);
    std::copy_n(observation, NUM_OBSERVATIONS,
                history + NUM_POLICY_INPUTS - NUM_OBSERVATIONS);
    rl_Inference inference(config.fel.model_path, config.fel.model_num_threads);
    inference.advanceNNsync_Walk(history, expectedAction);
    verifyMotorTargets(config.fel, expectedAction, *ctrlComp.lowCmd);

    for(int i = 0; i < NUM_ACTIONS; i++){
        lastAction[i] = std::clamp(expectedAction[i], -config.fel.action_clamp,
                                   config.fel.action_clamp);
    }
    state.run();
    rampedValue.ly -= config.fel.speed_ramp_step;
    rampedValue.rx += config.fel.speed_ramp_step;
    std::copy_n(history + NUM_OBSERVATIONS, NUM_POLICY_INPUTS - NUM_OBSERVATIONS,
                history);
    buildObservation(config, lowState, rampedValue, lastAction, observation);
    std::copy_n(observation, NUM_OBSERVATIONS,
                history + NUM_POLICY_INPUTS - NUM_OBSERVATIONS);
    inference.advanceNNsync_Walk(history, expectedAction);
    verifyMotorTargets(config.fel, expectedAction, *ctrlComp.lowCmd);

    for(int leg = 0; leg < 4; leg++){
        for(int joint = 0; joint < 3; joint++){
            const int index = leg * 3 + joint;
            requireNear(ctrlComp.lowCmd->motorCmd[index].Kp,
                        config.fel.sim_rl_gain.Kp[joint], "RL Kp mismatch");
            requireNear(ctrlComp.lowCmd->motorCmd[index].Kd,
                        config.fel.sim_rl_gain.Kd[joint], "RL Kd mismatch");
        }
    }
}

void testFsmTransitionTiming(const DeployConfig& config){
    auto *fake = new FakeIO();
    CtrlComponents ctrlComp(fake, &config);
    for(int i = 0; i < NUM_ACTIONS; i++){
        ctrlComp.lowState->motorState[i].q = config.fel.default_dof_pos[i] - 0.2F;
    }
    FSM fsm(&ctrlComp);
    require(fsm._currentState->_stateName == FSMStateName::PASSIVE,
            "FSM must initialize in passive");

    fake->userCommand = UserCommand::FIXED;
    fsm.run();
    require(fsm._currentState->_stateName == FSMStateName::PASSIVE,
            "FSM transition must be deferred until the next tick");
    fsm.run();
    require(fsm._currentState->_stateName == FSMStateName::FIXEDSTAND,
            "FSM did not enter fixed stand on the change tick");
    fsm.run();
    require(ctrlComp.lowCmd->motorCmd[0].q
            > ctrlComp.lowState->motorState[0].q,
            "Fixed stand did not start interpolation after entering");
}

int getActuatorId(const mjModel *model, int jointId){
    for(int actuatorId = 0; actuatorId < model->nu; actuatorId++){
        const int transmissionType = model->actuator_trntype[actuatorId];
        if((transmissionType == mjTRN_JOINT ||
            transmissionType == mjTRN_JOINTINPARENT) &&
           model->actuator_trnid[2 * actuatorId] == jointId){
            return actuatorId;
        }
    }
    return -1;
}

void testMujocoCommandFlow(const DeployConfig& config){
    char error[1024]{};
    std::unique_ptr<mjModel, decltype(&mj_deleteModel)> model(
        mj_loadXML(config.scene_path.c_str(), nullptr, error, sizeof(error)),
        &mj_deleteModel);
    require(model != nullptr, std::string("Failed to load MuJoCo test scene: ") + error);
    std::unique_ptr<mjData, decltype(&mj_deleteData)> data(mj_makeData(model.get()),
                                                          &mj_deleteData);
    require(data != nullptr, "Failed to allocate MuJoCo test data");
    mj_forward(model.get(), data.get());

    IOMujoco io(data.get(), model.get(), &config);
    LowlevelCmd lowCmd;
    LowlevelState lowState;
    io.recv(&lowState);
    for(int i = 0; i < NUM_ACTIONS; i++){
        lowCmd.motorCmd[i].q = lowState.motorState[i].q + 0.01F;
        lowCmd.motorCmd[i].dq = 0;
        lowCmd.motorCmd[i].Kp = 10;
        lowCmd.motorCmd[i].Kd = 0.5F;
    }
    io.sendRecv(&lowCmd, &lowState);
    io.send(&lowCmd);

    const int jointId = mj_name2id(model.get(), mjOBJ_JOINT, "FL_hip_joint");
    const int actuatorId = getActuatorId(model.get(), jointId);
    require(jointId >= 0 && actuatorId >= 0,
            "Failed to resolve the MuJoCo test actuator");
    requireNear(lowCmd.motorCmd[0].tau, 0.1F,
                "sendRecv did not compute the expected PD torque");
    requireNear(static_cast<float>(data->ctrl[actuatorId]), 0.1F,
                "send did not write the cached PD torque");

    lowCmd.motorCmd[0].tau = std::numeric_limits<float>::quiet_NaN();
    io.send(&lowCmd);
    requireNear(static_cast<float>(data->ctrl[actuatorId]), 0,
                "send did not reject a non-finite torque");
}

}

int main(){
    try{
        const std::filesystem::path sourceDir(DEPLOY_SOURCE_DIR);
        const DeployConfig config = DeployConfig::load(sourceDir / "configs/config.yaml");
        require(config.fel.model_input_size == NUM_POLICY_INPUTS,
                "Configured model input does not match deployment history");
        require(config.fel.model_output_size == NUM_ACTIONS,
                "Configured model output does not match deployment actions");
        requireNear(applyDeadzone(0.5F, config.joystick_deadzone),
                    (0.5F - config.joystick_deadzone) / (1 - config.joystick_deadzone),
                    "Deadzone reference calculation failed");
        testStateRlInterface(config);
        testFsmTransitionTiming(config);
        testMujocoCommandFlow(config);
        std::cout << "deploy interface tests passed" << std::endl;
        return 0;
    }
    catch(const std::exception& error){
        std::cerr << "deploy interface tests failed: " << error.what() << std::endl;
        return 1;
    }
}
