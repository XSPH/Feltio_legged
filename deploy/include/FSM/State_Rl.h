#ifndef STATE_RL_H
#define STATE_RL_H

#include <filesystem>
#include <memory>

#include <Eigen/Geometry>

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
    void stateMachine_fel();
    void speed_limit();
    void mnnInference_fel();
    float applyDeadzone(float value) const;
    Eigen::Vector3f quat_rotate_inverse(const Eigen::Quaternionf& quaternion,
                                        const Eigen::Vector3f& vector) const;

    std::shared_ptr<rl_Inference> rlptr = nullptr;

    float obs_fel[NUM_OBSERVATIONS];
    float obs_history_fel[NUM_POLICY_INPUTS];
    float action_cmd_fel[NUM_ACTIONS];
    float last_action_cmd_fel[NUM_ACTIONS];
    float proj_gravity[3];

    std::filesystem::path current_legged_model_path;
};

#endif
