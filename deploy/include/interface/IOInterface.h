#ifndef IOINTERFACE_H
#define IOINTERFACE_H

#include <memory>

#include "interface/CmdPanel.h"
#include "message/LowlevelCmd.h"
#include "message/LowlevelState.h"

class IOInterface{
public:
    IOInterface() = default;
    virtual ~IOInterface() = default;

    virtual void sendRecv(LowlevelCmd *cmd, LowlevelState *state) = 0;
    virtual void send(LowlevelCmd *cmd) = 0;
    virtual void recv(LowlevelState *state) = 0;

    void zeroCmdPanel(){
        if(cmdPanel){
            cmdPanel->setZero();
        }
    }

    void setUserCommand(UserCommand command){
        if(cmdPanel){
            cmdPanel->setUserCommand(command);
        }
    }

    std::unique_ptr<CmdPanel> cmdPanel;
};

#endif
