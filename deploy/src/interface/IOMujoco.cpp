#include "interface/IOMujoco.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>

namespace{

const char *jointNames[12] = {
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"
};

int getMujocoId(const mjModel *model, mjtObj type, const char *name){
    int id = mj_name2id(model, type, name);
    if(id < 0){
        throw std::runtime_error(std::string("MuJoCo object not found: ") + name);
    }
    return id;
}

int getRootJointId(const mjModel *model){
    int rootJointId = mj_name2id(model, mjOBJ_JOINT, "root");
    if(rootJointId >= 0){
        if(model->jnt_type[rootJointId] != mjJNT_FREE){
            throw std::runtime_error("MuJoCo joint 'root' must be a free joint");
        }
        return rootJointId;
    }
    for(int jointId = 0; jointId < model->njnt; jointId++){
        if(model->jnt_type[jointId] == mjJNT_FREE){
            return jointId;
        }
    }
    throw std::runtime_error("MuJoCo free root joint not found");
}

int getJointActuatorId(const mjModel *model, int jointId, const char *jointName){
    for(int actuatorId = 0; actuatorId < model->nu; actuatorId++){
        int transmissionType = model->actuator_trntype[actuatorId];
        if((transmissionType == mjTRN_JOINT || transmissionType == mjTRN_JOINTINPARENT) &&
           model->actuator_trnid[2 * actuatorId] == jointId){
            return actuatorId;
        }
    }
    throw std::runtime_error(std::string("MuJoCo actuator not found for joint: ") + jointName);
}

}

IOMujoco::IOMujoco(mjData *data, mjModel *model, const DeployConfig *config)
    : _data(data), _model(model), _rootQposAddr(-1), _rootDofAddr(-1){
    if(_data == nullptr || _model == nullptr || config == nullptr){
        throw std::runtime_error("IOMujoco received a null pointer");
    }

    cmdPanel = std::make_unique<WirelessHandle>(config->joystick_device);
    for(int i = 0; i < 12; i++){
        int jointId = getMujocoId(_model, mjOBJ_JOINT, jointNames[i]);
        _qposAddr[i] = _model->jnt_qposadr[jointId];
        _dofAddr[i] = _model->jnt_dofadr[jointId];
        _actuatorId[i] = getJointActuatorId(_model, jointId, jointNames[i]);
    }

    int rootJointId = getRootJointId(_model);
    _rootQposAddr = _model->jnt_qposadr[rootJointId];
    _rootDofAddr = _model->jnt_dofadr[rootJointId];
}

void IOMujoco::sendRecv(LowlevelCmd *cmd, LowlevelState *state){
    recv(state);
    for(int i = 0; i < 12; i++){
        cmd->motorCmd[i].tau = cmd->motorCmd[i].Kp
                             * (cmd->motorCmd[i].q - state->motorState[i].q)
                             + cmd->motorCmd[i].Kd
                             * (cmd->motorCmd[i].dq - state->motorState[i].dq);
    }

    state->userCmd = cmdPanel->getUserCmd();
    const UserValue joystickValue = cmdPanel->getUserValue();
    state->userValue.lx = std::clamp(joystickValue.lx + _keyboardValue.lx,
                                    -1.0f, 1.0f);
    state->userValue.ly = std::clamp(joystickValue.ly + _keyboardValue.ly,
                                    -1.0f, 1.0f);
    state->userValue.rx = std::clamp(joystickValue.rx + _keyboardValue.rx,
                                    -1.0f, 1.0f);
    state->userValue.ry = std::clamp(joystickValue.ry + _keyboardValue.ry,
                                    -1.0f, 1.0f);
}

void IOMujoco::setKeyboardValue(const UserValue& value){
    _keyboardValue = value;
}

void IOMujoco::recv(LowlevelState *state){
    for(int i = 0; i < 12; i++){
        state->motorState[i].q = static_cast<float>(_data->qpos[_qposAddr[i]]);
        state->motorState[i].dq = static_cast<float>(_data->qvel[_dofAddr[i]]);
        state->motorState[i].tauEst = static_cast<float>(_data->qfrc_actuator[_dofAddr[i]]);
    }

    for(int i = 0; i < 4; i++){
        state->imu.quaternion[i] = static_cast<float>(_data->qpos[_rootQposAddr + 3 + i]);
    }

    // Free-joint angular qvel is in the body frame. mj_objectVelocity with
    // local orientation instead uses the principal-inertia frame.
    for(int i = 0; i < 3; i++){
        state->imu.gyroscope[i] = static_cast<float>(_data->qvel[_rootDofAddr + 3 + i]);
    }
}

void IOMujoco::send(LowlevelCmd *cmd){
    for(int i = 0; i < 12; i++){
        float torque = cmd->motorCmd[i].tau;

        if(!std::isfinite(torque)){
            torque = 0;
        }

        int actuatorId = _actuatorId[i];
        if(_model->actuator_ctrllimited[actuatorId]){
            const mjtNum *range = _model->actuator_ctrlrange + 2 * actuatorId;
            torque = std::clamp(torque, static_cast<float>(range[0]),
                                static_cast<float>(range[1]));
        }
        _data->ctrl[actuatorId] = torque;
    }
}
