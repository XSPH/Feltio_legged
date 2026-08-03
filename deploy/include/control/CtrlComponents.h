#ifndef CTRLCOMPONENTS_H
#define CTRLCOMPONENTS_H

#include "config/DeployConfig.h"
#include "interface/IOInterface.h"
#include "message/LowlevelCmd.h"
#include "message/LowlevelState.h"

struct CtrlComponents{
public:
    CtrlComponents(IOInterface *ioInter, const DeployConfig *config)
        : ioInter(ioInter), config(config){
        lowCmd = new LowlevelCmd();
        lowState = new LowlevelState();
    }

    ~CtrlComponents(){
        delete lowCmd;
        delete lowState;
        delete ioInter;
    }

    void sendRecv(){
        ioInter->sendRecv(lowCmd, lowState);
    }

    void send(){
        ioInter->send(lowCmd, lowState);
    }

    void setUserCommand(UserCommand command){
        ioInter->setUserCommand(command);
    }

    void zeroCmdPanel(){
        ioInter->zeroCmdPanel();
    }

    LowlevelCmd *lowCmd;
    LowlevelState *lowState;
    IOInterface *ioInter;
    const DeployConfig *config;
};

#endif
