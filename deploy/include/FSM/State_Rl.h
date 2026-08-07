#ifndef STATE_RL_H
#define STATE_RL_H

#include <memory>

#include "FSM/FSMState.h"
#include "control/rl_Inference.h"

class State_Rl : public FSMState{
public:
    State_Rl(CtrlComponents *ctrlComp);
    void enter();
    void run();
    void exit();
    FSMStateName checkChange();

private:
    void getObservation();
    void mnnInference();
    float applyDeadzone(float value);

    std::shared_ptr<rl_Inference> rlptr;
    float obs[NUM_OBSERVATIONS];
    float obsHistory[NUM_POLICY_INPUTS];
    float actionCmd[NUM_ACTIONS];
    float lastAction[NUM_ACTIONS];
};

#endif
