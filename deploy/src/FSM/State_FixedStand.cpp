#include "FSM/State_FixedStand.h"

#include <algorithm>

State_FixedStand::State_FixedStand(CtrlComponents *ctrlComp)
    : FSMState(ctrlComp, FSMStateName::FIXEDSTAND, "fixed stand"), _percent(0){
    for(int i = 0; i < 12; i++){
        _startPos[i] = 0;
        _targetPos[i] = 0;
    }
}

void State_FixedStand::enter(){
    _percent = 0;
    const FelConfig& cfg = _config->fel;
    _ctrlComp->clearVelocityCommands();
    for(int i = 0; i < 4; i++){
        _lowCmd->setLegGains(i, cfg.sim_stance_gain);
        _lowCmd->setZeroDq(i);
        _lowCmd->setZeroTau(i);
    }
    for(int i = 0; i < 12; i++){
        _startPos[i] = _lowState->motorState[i].q;
        _targetPos[i] = cfg.default_dof_pos[i];
    }
}

void State_FixedStand::run(){
    _ctrlComp->clearVelocityCommands();
    float controlDt = static_cast<float>(_config->simulation_timestep
                                         * _config->simulation_decimation);
    _percent += controlDt / _config->fel.stand_duration;
    _percent = std::min(_percent, 1.0f);

    for(int i = 0; i < 12; i++){
        _lowCmd->motorCmd[i].q = (1 - _percent) * _startPos[i]
                               + _percent * _targetPos[i];
        _lowCmd->motorCmd[i].dq = 0;
    }
}

void State_FixedStand::exit(){
    _percent = 0;
}

FSMStateName State_FixedStand::checkChange(){
    if(_lowState->userCmd == UserCommand::PASS){
        return FSMStateName::PASSIVE;
    }
    else if(_lowState->userCmd == UserCommand::RL){
        return FSMStateName::Rl;
    }
    return FSMStateName::FIXEDSTAND;
}
