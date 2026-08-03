#ifndef IOINTERFACE_H
#define IOINTERFACE_H

#include "interface/CmdPanel.h"
#include "message/LowlevelCmd.h"
#include "message/LowlevelState.h"

class IOInterface{
public:
    IOInterface(): cmdPanel(nullptr){}
    virtual ~IOInterface(){
        delete cmdPanel;
    }

    virtual void sendRecv(LowlevelCmd *cmd, LowlevelState *state) = 0;
    virtual void send(LowlevelCmd *cmd, LowlevelState *state) = 0;
    virtual void recv(LowlevelState *state) = 0;

    void zeroCmdPanel(){
        cmdPanel->setZero();
    }

    void setUserCommand(UserCommand command){
        cmdPanel->setUserCommand(command);
    }

    CmdPanel *cmdPanel;
};

#endif
