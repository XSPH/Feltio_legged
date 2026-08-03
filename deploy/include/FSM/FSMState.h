#ifndef FSMSTATE_H
#define FSMSTATE_H

#include <string>

#include "common/enumClass.h"
#include "control/CtrlComponents.h"
#include "interface/CmdPanel.h"
#include "message/LowlevelCmd.h"
#include "message/LowlevelState.h"

class FSMState{
public:
    FSMState(CtrlComponents *ctrlComp, FSMStateName stateName,
             std::string stateNameString);
    virtual ~FSMState(){}

    virtual void enter() = 0;
    virtual void run() = 0;
    virtual void exit() = 0;
    virtual FSMStateName checkChange() = 0;

    FSMStateName _stateName;
    std::string _stateNameString;

protected:
    CtrlComponents *_ctrlComp;
    LowlevelCmd *_lowCmd;
    LowlevelState *_lowState;
    const DeployConfig *_config;
    UserValue _userValue;
};

#endif

