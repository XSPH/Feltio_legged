#ifndef LOWLEVELCMD_H
#define LOWLEVELCMD_H

#include "config/DeployConfig.h"

struct MotorCmd_user{
    float q;
    float dq;
    float tau;
    float Kp;
    float Kd;

    MotorCmd_user(){
        q = 0;
        dq = 0;
        tau = 0;
        Kp = 0;
        Kd = 0;
    }
};

struct LowlevelCmd{
    MotorCmd_user motorCmd[12];

    void setLegGains(int legId, const LegGains& gains){
        for(int i = 0; i < 3; i++){
            motorCmd[legId * 3 + i].Kp = gains.Kp[i];
            motorCmd[legId * 3 + i].Kd = gains.Kd[i];
        }
    }

    void setZeroDq(int legId){
        for(int i = 0; i < 3; i++){
            motorCmd[legId * 3 + i].dq = 0;
        }
    }

    void setZeroTau(int legId){
        for(int i = 0; i < 3; i++){
            motorCmd[legId * 3 + i].tau = 0;
        }
    }

    void setPassive(){
        for(int i = 0; i < 12; i++){
            motorCmd[i] = MotorCmd_user();
        }
    }
};

#endif
