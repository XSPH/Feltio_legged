#ifndef LOWLEVELCMD_H
#define LOWLEVELCMD_H

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

    void setGain(float kp, float kd){
        for(int i = 0; i < 12; i++){
            motorCmd[i].Kp = kp;
            motorCmd[i].Kd = kd;
        }
    }

    void setStanceGain(){
        for(int i = 0; i < 4; i++){
            motorCmd[i * 3].Kp = 30;
            motorCmd[i * 3].Kd = 0.75;
            motorCmd[i * 3 + 1].Kp = 50;
            motorCmd[i * 3 + 1].Kd = 1.25;
            motorCmd[i * 3 + 2].Kp = 60;
            motorCmd[i * 3 + 2].Kd = 1.5;
        }
    }

    void setPassive(){
        for(int i = 0; i < 12; i++){
            motorCmd[i] = MotorCmd_user();
        }
    }
};

#endif
