#ifndef CONTROLFRAME_H
#define CONTROLFRAME_H

#include "FSM/FSM.h"
#include "control/CtrlComponents.h"

class ControlFrame{
public:
    ControlFrame(CtrlComponents *ctrlComp);
    ~ControlFrame();

    void run();
    void reset();

    FSM *_FSMController;
    CtrlComponents *_ctrlComp;
};

#endif

