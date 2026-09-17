#ifndef CTRLCOMPONENTS_H
#define CTRLCOMPONENTS_H

#include <array>
#include <memory>

#include "config/DeployConfig.h"
#include "interface/IOInterface.h"
#include "message/LowlevelCmd.h"
#include "message/LowlevelState.h"

struct CtrlComponents{
public:
    CtrlComponents(IOInterface *ioInter, const DeployConfig *config)
        : ioInter(ioInter), config(config){
        lowCmdOwner = std::make_unique<LowlevelCmd>();
        lowStateOwner = std::make_unique<LowlevelState>();
        ioInterOwner.reset(ioInter);
        lowCmd = lowCmdOwner.get();
        lowState = lowStateOwner.get();
    }

    ~CtrlComponents() = default;

    void sendRecv(){
        ioInter->sendRecv(lowCmd, lowState);
    }

    void send(){
        ioInter->send(lowCmd);
    }

    void recv(){
        ioInter->recv(lowState);
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
    std::array<float, 3> targetVelocityCommand{};
    std::array<float, 3> appliedVelocityCommand{};

    void clearVelocityCommands(){
        targetVelocityCommand.fill(0.0F);
        appliedVelocityCommand.fill(0.0F);
    }

private:
    std::unique_ptr<LowlevelCmd> lowCmdOwner;
    std::unique_ptr<LowlevelState> lowStateOwner;
    std::unique_ptr<IOInterface> ioInterOwner;
};

#endif
