#ifndef LOWLEVELSTATE_H
#define LOWLEVELSTATE_H

#include "common/enumClass.h"
#include "interface/CmdPanel.h"

struct MotorState{
    float q;
    float dq;
    float tauEst;

    MotorState(){
        q = 0;
        dq = 0;
        tauEst = 0;
    }
};

struct IMU{
    float quaternion[4];   // w, x, y, z
    float gyroscope[3];   // body frame

    IMU(){
        quaternion[0] = 1;
        for(int i = 1; i < 4; i++){
            quaternion[i] = 0;
        }
        for(int i = 0; i < 3; i++){
            gyroscope[i] = 0;
        }
    }
};

struct LowlevelState{
    IMU imu;
    MotorState motorState[12];
    UserCommand userCmd;
    UserValue userValue;

    LowlevelState(): userCmd(UserCommand::PASS){}
};

#endif
