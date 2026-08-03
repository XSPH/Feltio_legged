#include "FSM/FSM.h"

#include <iostream>
#include <stdexcept>

FSM::FSM(CtrlComponents *ctrlComp): _ctrlComp(ctrlComp){
    _stateList.passive = new State_Passive(_ctrlComp);
    _stateList.fixedStand = new State_FixedStand(_ctrlComp);
    _stateList.rl = new State_Rl(_ctrlComp);
    initialize();
}

FSM::~FSM(){
    _stateList.deletePtr();
}

void FSM::initialize(){
    _currentState = _stateList.passive;
    _nextState = _currentState;
    _mode = FSMMode::NORMAL;
    _currentState->enter();
    std::cout << "[FSM] reset -> passive" << std::endl;
}

void FSM::reset(){
    if(_currentState != nullptr){
        _currentState->exit();
    }
    _ctrlComp->setUserCommand(UserCommand::PASS);
    initialize();
}

void FSM::run(){
    _ctrlComp->sendRecv();

    if(_mode == FSMMode::NORMAL){
        _nextStateName = _currentState->checkChange();
        if(_nextStateName != _currentState->_stateName){
            _mode = FSMMode::CHANGE;
            _nextState = getNextState(_nextStateName);
        }
    }

    if(_mode == FSMMode::CHANGE){
        std::cout << "[FSM] " << _currentState->_stateNameString
                  << " -> " << _nextState->_stateNameString << std::endl;
        _currentState->exit();
        _currentState = _nextState;
        _currentState->enter();
        _mode = FSMMode::NORMAL;
    }

    _currentState->run();
    _ctrlComp->send();
}

FSMState *FSM::getNextState(FSMStateName stateName){
    switch(stateName){
    case FSMStateName::PASSIVE:
        return _stateList.passive;
    case FSMStateName::FIXEDSTAND:
        return _stateList.fixedStand;
    case FSMStateName::Rl:
        return _stateList.rl;
    default:
        throw std::runtime_error("Invalid FSM state");
    }
}

