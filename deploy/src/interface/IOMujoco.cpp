#include "interface/IOMujoco.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace{

const char *jointNames[12] = {
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"
};

const char *actuatorNames[12] = {
    "FL_hip", "FL_thigh", "FL_calf",
    "FR_hip", "FR_thigh", "FR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
    "RR_hip", "RR_thigh", "RR_calf"
};

int getMujocoId(const mjModel *model, mjtObj type, const char *name){
    int id = mj_name2id(model, type, name);
    if(id < 0){
        throw std::runtime_error(std::string("MuJoCo object not found: ") + name);
    }
    return id;
}

}

IOMujoco::IOMujoco(mjData *data, mjModel *model, const DeployConfig *config)
    : _data(data), _model(model), _baseBodyId(-1), _rootQposAddr(-1){
    if(_data == nullptr || _model == nullptr || config == nullptr){
        throw std::runtime_error("IOMujoco received a null pointer");
    }

    cmdPanel = new WirelessHandle(config->joystick_device);
    for(int i = 0; i < 12; i++){
        int jointId = getMujocoId(_model, mjOBJ_JOINT, jointNames[i]);
        _qposAddr[i] = _model->jnt_qposadr[jointId];
        _dofAddr[i] = _model->jnt_dofadr[jointId];
        _actuatorId[i] = getMujocoId(_model, mjOBJ_ACTUATOR, actuatorNames[i]);
    }

    _baseBodyId = getMujocoId(_model, mjOBJ_BODY, "base");
    int rootJointId = getMujocoId(_model, mjOBJ_JOINT, "root");
    if(_model->jnt_type[rootJointId] != mjJNT_FREE){
        throw std::runtime_error("MuJoCo joint 'root' must be a free joint");
    }
    _rootQposAddr = _model->jnt_qposadr[rootJointId];
}

void IOMujoco::sendRecv(LowlevelCmd *cmd, LowlevelState *state){
    (void)cmd;
    recv(state);
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

    mjtNum velocity[6] = {0};
    mj_objectVelocity(_model, _data, mjOBJ_BODY, _baseBodyId, velocity, 1);
    for(int i = 0; i < 3; i++){
        state->imu.gyroscope[i] = static_cast<float>(velocity[i]);
        state->imu.line[i] = static_cast<float>(velocity[i + 3]);
    }

    state->userCmd = cmdPanel->getUserCmd();
    state->userValue = cmdPanel->getUserValue();
}

void IOMujoco::send(LowlevelCmd *cmd, LowlevelState *state){
    for(int i = 0; i < 12; i++){
        float torque = cmd->motorCmd[i].tau
                     + cmd->motorCmd[i].Kp * (cmd->motorCmd[i].q - state->motorState[i].q)
                     + cmd->motorCmd[i].Kd * (cmd->motorCmd[i].dq - state->motorState[i].dq);

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
