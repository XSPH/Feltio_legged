#include "FSM/State_Rl.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <Eigen/Geometry>

State_Rl::State_Rl(CtrlComponents *ctrlComp)
    : FSMState(ctrlComp, FSMStateName::Rl, "rl"){
    rlptr = std::make_shared<rl_Inference>(_config->model_path,
                                           _config->model_num_threads);
    for(int i = 0; i < NUM_OBSERVATIONS; i++){
        obs[i] = 0;
    }
    for(int i = 0; i < NUM_ACTIONS; i++){
        actionCmd[i] = 0;
        lastAction[i] = 0;
    }
}

void State_Rl::enter(){
    _lowCmd->setGain(_config->stiffness, _config->damping);
    for(int i = 0; i < NUM_ACTIONS; i++){
        actionCmd[i] = 0;
        lastAction[i] = 0;
    }
}

void State_Rl::run(){
    _userValue = _lowState->userValue;
    getObservation();
    mnnInference();
}

float State_Rl::applyDeadzone(float value){
    if(std::abs(value) <= _config->joystick_deadzone){
        return 0;
    }
    float magnitude = (std::abs(value) - _config->joystick_deadzone)
                      / (1 - _config->joystick_deadzone);
    return std::copysign(magnitude, value);
}

void State_Rl::getObservation(){
    // 0~2: base angular velocity
    for(int i = 0; i < 3; i++){
        obs[i] = _lowState->imu.gyroscope[i] * _config->angular_velocity_scale;
    }

    // 3~5: projected gravity
    Eigen::Quaternionf baseQuat(_lowState->imu.quaternion[0],
                                _lowState->imu.quaternion[1],
                                _lowState->imu.quaternion[2],
                                _lowState->imu.quaternion[3]);
    if(baseQuat.norm() < 1e-6f){
        throw std::runtime_error("Invalid base quaternion from MuJoCo");
    }
    baseQuat.normalize();
    Eigen::Vector3f gravity = baseQuat.conjugate()
                            * Eigen::Vector3f(0, 0, -1);
    for(int i = 0; i < 3; i++){
        obs[i + 3] = gravity[i];
    }

    // 6~8: vx, vy and yaw commands from the gamepad
    float commandX = -applyDeadzone(_userValue.ly) * _config->max_linear_x;
    float commandY = -applyDeadzone(_userValue.lx) * _config->max_linear_y;
    float commandYaw = -applyDeadzone(_userValue.rx) * _config->max_angular_z;
    obs[6] = commandX * _config->command_linear_scale;
    obs[7] = commandY * _config->command_linear_scale;
    obs[8] = commandYaw * _config->command_yaw_scale;

    // 9~20: joint position, 21~32: joint velocity, 33~44: last action
    for(int i = 0; i < 12; i++){
        obs[i + 9] = (_lowState->motorState[i].q
                       - _config->default_joint_angles[i])
                      * _config->joint_position_scale;
        obs[i + 21] = _lowState->motorState[i].dq
                      * _config->joint_velocity_scale;
        obs[i + 33] = lastAction[i];
    }

    for(int i = 0; i < NUM_OBSERVATIONS; i++){
        if(!std::isfinite(obs[i])){
            throw std::runtime_error("Non-finite value in policy observation");
        }
        obs[i] = std::clamp(obs[i], -_config->clip_observations,
                            _config->clip_observations);
    }
}

void State_Rl::mnnInference(){
    // One feed-forward actor: 45 observations -> 12 actions.
    rlptr->advanceNNsync(obs, actionCmd);

    for(int i = 0; i < NUM_ACTIONS; i++){
        if(!std::isfinite(actionCmd[i])){
            throw std::runtime_error("Policy returned a non-finite action");
        }
        actionCmd[i] = std::clamp(actionCmd[i], -_config->clip_actions,
                                  _config->clip_actions);
        const float jointActionScale =
            _config->action_scale * (i % 3 == 0 ? _config->hip_reduction : 1.0F);
        _lowCmd->motorCmd[i].q = _config->default_joint_angles[i]
                               + actionCmd[i] * jointActionScale;
        _lowCmd->motorCmd[i].dq = 0;
        _lowCmd->motorCmd[i].tau = 0;
        lastAction[i] = actionCmd[i];
    }
}

void State_Rl::exit(){
    for(int i = 0; i < NUM_ACTIONS; i++){
        lastAction[i] = 0;
    }
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
