#ifndef FSM_H
#define FSM_H

#include "FSM/FSMState.h"
#include "FSM/State_FixedStand.h"
#include "FSM/State_Passive.h"
#include "FSM/State_Rl.h"

struct FSMStateList{
    State_Passive *passive;
    State_FixedStand *fixedStand;
    State_Rl *rl;

    void deletePtr(){
        delete passive;
        delete fixedStand;
        delete rl;
    }
};

class FSM{
public:
    FSM(CtrlComponents *ctrlComp);
    ~FSM();

    void initialize();
    void run();
    void reset();
    FSMState *getNextState(FSMStateName stateName);

    FSMState *_currentState;

private:
    CtrlComponents *_ctrlComp;
    FSMState *_nextState;
    FSMStateName _nextStateName;
    FSMStateList _stateList;
    FSMMode _mode;
};

#endif

