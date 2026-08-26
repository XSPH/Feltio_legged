#include "FSM/State_Rl.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>

State_Rl::State_Rl(CtrlComponents *ctrlComp)
    : FSMState(ctrlComp, FSMStateName::Rl, "rl model"){}

void State_Rl::enter(){
    const FelConfig& cfg = _config->fel;
    for(int i = 0; i < 4; i++){
        _lowCmd->setLegGains(i, cfg.sim_rl_gain);
        _lowCmd->setZeroDq(i);
        _lowCmd->setZeroTau(i);
    }

    std::memset(obs_fel, 0, sizeof(obs_fel));
    std::memset(obs_history_fel, 0, sizeof(obs_history_fel));
    std::memset(action_cmd_fel, 0, sizeof(action_cmd_fel));
    std::memset(last_action_cmd_fel, 0, sizeof(last_action_cmd_fel));
    std::memset(proj_gravity, 0, sizeof(proj_gravity));
    _userValue.setZero();
    current_legged_model_path.clear();
}

void State_Rl::run(){
    speed_limit();
    stateMachine_fel();
    mnnInference_fel();
}

void State_Rl::exit(){
    _userValue.setZero();
    _ctrlComp->zeroCmdPanel();
}

FSMStateName State_Rl::checkChange(){
    if(_lowState->userCmd == UserCommand::FIXED){
        return FSMStateName::FIXEDSTAND;
    }
    else if(_lowState->userCmd == UserCommand::PASS){
        return FSMStateName::PASSIVE;
    }
    return FSMStateName::Rl;
}

void State_Rl::stateMachine_fel(){
    const FelConfig& cfg = _config->fel;
    if(rlptr == nullptr || current_legged_model_path != cfg.model_path){
        std::cout << "[State_Rl] loading fel policy model: "
                  << cfg.model_path << std::endl;
        rlptr = std::make_shared<rl_Inference>(cfg.model_path,
                                               cfg.model_num_threads);
        current_legged_model_path = cfg.model_path;
    }
}

void State_Rl::speed_limit(){
    const FelConfig& cfg = _config->fel;
    UserValue target;
    target.lx = applyDeadzone(_lowState->userValue.lx);
    target.ly = applyDeadzone(_lowState->userValue.ly);
    target.rx = applyDeadzone(_lowState->userValue.rx);
    target.ry = applyDeadzone(_lowState->userValue.ry);
    const auto ramp = [&cfg](float& current, float requested){
        if(std::abs(requested - current) > cfg.speed_ramp_step){
            current += requested > current ? cfg.speed_ramp_step
                                           : -cfg.speed_ramp_step;
        }
        else{
            current = requested;
        }
    };
    ramp(_userValue.lx, target.lx);
    ramp(_userValue.ly, target.ly);
    ramp(_userValue.rx, target.rx);
    ramp(_userValue.ry, target.ry);
}

void State_Rl::mnnInference_fel(){
    const FelConfig& cfg = _config->fel;

    // Project world gravity into the body frame using MuJoCo's wxyz quaternion.
    Eigen::Quaternionf baseQuat(_lowState->imu.quaternion[0],
                                _lowState->imu.quaternion[1],
                                _lowState->imu.quaternion[2],
                                _lowState->imu.quaternion[3]);
    if(baseQuat.norm() < 1.0e-6F){
        throw std::runtime_error("Invalid base quaternion from MuJoCo");
    }
    baseQuat.normalize();
    const Eigen::Vector3f gravity = quat_rotate_inverse(
        baseQuat, Eigen::Vector3f(0.0F, 0.0F, -1.0F));
    for(int i = 0; i < 3; i++){
        proj_gravity[i] = gravity[i];
        obs_fel[i] = _lowState->imu.gyroscope[i] * cfg.scale_ang_vel;
        obs_fel[i + 3] = proj_gravity[i];
    }

    // Velocity commands use the same signs and scales as the training observations.
    obs_fel[6] = -_userValue.ly * _config->max_linear_x
               * cfg.scale_lin_vel_x;
    obs_fel[7] = -_userValue.lx * _config->max_linear_y
               * cfg.scale_lin_vel_y;
    obs_fel[8] = -_userValue.rx * _config->max_angular_z
               * cfg.scale_ang_vel;

    for(int i = 0; i < NUM_ACTIONS; i++){
        obs_fel[i + 9] = (_lowState->motorState[i].q
                          - cfg.default_dof_pos[i]) * cfg.scale_dof_pos;
        obs_fel[i + 21] = _lowState->motorState[i].dq * cfg.scale_dof_vel;
        obs_fel[i + 33] = last_action_cmd_fel[i];
    }
    for(float& value : obs_fel){
        if(!std::isfinite(value)){
            throw std::runtime_error("Non-finite value in policy observation");
        }
        value = std::clamp(value, -cfg.observation_clamp,
                           cfg.observation_clamp);
    }

    // Roll the five-frame history from oldest to newest, then run the fused
    // student encoder and actor model.
    for(int i = 0; i < NUM_POLICY_INPUTS - NUM_OBSERVATIONS; i++){
        obs_history_fel[i] = obs_history_fel[i + NUM_OBSERVATIONS];
    }
    std::copy_n(obs_fel, NUM_OBSERVATIONS,
                obs_history_fel + NUM_POLICY_INPUTS - NUM_OBSERVATIONS);

    rlptr->advanceNNsync_Walk(obs_history_fel, action_cmd_fel);

    // Preserve the clipped pre-reduction action for observation history, but
    // apply the configured hip reduction only to the outgoing position target.
    float action_flt[NUM_ACTIONS];
    for(int i = 0; i < NUM_ACTIONS; i++){
        if(!std::isfinite(action_cmd_fel[i])){
            throw std::runtime_error("Policy returned a non-finite action");
        }
        action_cmd_fel[i] = std::clamp(action_cmd_fel[i], -cfg.action_clamp,
                                       cfg.action_clamp);
        action_flt[i] = action_cmd_fel[i];
        last_action_cmd_fel[i] = action_cmd_fel[i];
    }
    for(int leg = 0; leg < 4; leg++){
        action_flt[leg * 3] *= cfg.hip_reduction;
    }
    for(int i = 0; i < NUM_ACTIONS; i++){
        _lowCmd->motorCmd[i].q = action_flt[i] * cfg.actions_scale
                               + cfg.default_dof_pos[i];
        _lowCmd->motorCmd[i].dq = 0;
    }
}

float State_Rl::applyDeadzone(float value) const{
    if(std::abs(value) <= _config->joystick_deadzone){
        return 0;
    }
    const float magnitude = (std::abs(value) - _config->joystick_deadzone)
                          / (1 - _config->joystick_deadzone);
    return std::copysign(magnitude, value);
}

Eigen::Vector3f State_Rl::quat_rotate_inverse(
    const Eigen::Quaternionf& quaternion, const Eigen::Vector3f& vector) const{
    return quaternion.conjugate() * vector;
}
