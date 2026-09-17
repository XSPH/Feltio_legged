#include "FSM/State_Passive.h"

State_Passive::State_Passive(CtrlComponents *ctrlComp)
    : FSMState(ctrlComp, FSMStateName::PASSIVE, "passive"){}

void State_Passive::enter(){
    _lowCmd->setPassive();
    _ctrlComp->zeroCmdPanel();
    _ctrlComp->clearVelocityCommands();
}

void State_Passive::run(){
    _lowCmd->setPassive();
    _ctrlComp->clearVelocityCommands();
}

void State_Passive::exit(){}

FSMStateName State_Passive::checkChange(){
    if(_lowState->userCmd == UserCommand::FIXED){
        return FSMStateName::FIXEDSTAND;
    }
    return FSMStateName::PASSIVE;
}
