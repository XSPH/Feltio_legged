#include "control/ControlFrame.h"

ControlFrame::ControlFrame(CtrlComponents *ctrlComp): _ctrlComp(ctrlComp){
    _FSMController = new FSM(_ctrlComp);
}

ControlFrame::~ControlFrame(){
    delete _FSMController;
}

void ControlFrame::run(){
    _FSMController->run();
}

void ControlFrame::reset(){
    _FSMController->reset();
}
