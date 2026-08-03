#include "FSM/FSMState.h"

FSMState::FSMState(CtrlComponents *ctrlComp, FSMStateName stateName,
                   std::string stateNameString)
    : _stateName(stateName), _stateNameString(stateNameString), _ctrlComp(ctrlComp){
    _lowCmd = _ctrlComp->lowCmd;
    _lowState = _ctrlComp->lowState;
    _config = _ctrlComp->config;
}

