#ifndef IOMUJOCO_H
#define IOMUJOCO_H

#include <array>

#include <mujoco/mujoco.h>

#include "config/DeployConfig.h"
#include "interface/IOInterface.h"
#include "interface/WirelessHandle.h"

class IOMujoco : public IOInterface{
public:
    IOMujoco(mjData *data, mjModel *model, const DeployConfig *config);
    ~IOMujoco(){}

    void sendRecv(LowlevelCmd *cmd, LowlevelState *state);
    void send(LowlevelCmd *cmd, LowlevelState *state);
    void recv(LowlevelState *state);

private:
    mjData *_data;
    mjModel *_model;
    std::array<int, 12> _qposAddr;
    std::array<int, 12> _dofAddr;
    std::array<int, 12> _actuatorId;
    int _baseBodyId;
    int _rootQposAddr;
};

#endif

