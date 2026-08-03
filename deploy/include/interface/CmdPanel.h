#ifndef CMDPANEL_H
#define CMDPANEL_H

#include <mutex>

#include "common/enumClass.h"

struct UserValue{
    float lx;
    float ly;
    float rx;
    float ry;

    UserValue(){
        setZero();
    }

    void setZero(){
        lx = 0;
        ly = 0;
        rx = 0;
        ry = 0;
    }
};

class CmdPanel{
public:
    CmdPanel(): userCmd(UserCommand::PASS){}
    virtual ~CmdPanel(){}

    UserCommand getUserCmd(){
        std::lock_guard<std::mutex> lock(_mutex);
        return userCmd;
    }

    UserValue getUserValue(){
        std::lock_guard<std::mutex> lock(_mutex);
        return userValue;
    }

    void setUserCommand(UserCommand command){
        std::lock_guard<std::mutex> lock(_mutex);
        userCmd = command;
    }

    void setPassive(){
        setUserCommand(UserCommand::PASS);
    }

    void setZero(){
        std::lock_guard<std::mutex> lock(_mutex);
        userValue.setZero();
    }

protected:
    void setUserValue(const UserValue& value){
        std::lock_guard<std::mutex> lock(_mutex);
        userValue = value;
    }

    UserCommand userCmd;
    UserValue userValue;
    std::mutex _mutex;
};

#endif

