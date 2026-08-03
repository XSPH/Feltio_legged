#ifndef RL_INFERENCE_H
#define RL_INFERENCE_H

#include <filesystem>

#include "MNN/Interpreter.hpp"
#include "MNN/Tensor.hpp"

#define NUM_OBSERVATIONS 48
#define NUM_ACTIONS 12

class rl_Inference{
public:
    rl_Inference(const std::filesystem::path& modelPath, int numThreads);
    ~rl_Inference();

    void advanceNNsync(const float observation[NUM_OBSERVATIONS],
                       float actionCmd[NUM_ACTIONS]);

private:
    MNN::Interpreter *_net;
    MNN::Session *_session;
    MNN::Tensor *_inputTensor;
    MNN::Tensor *_outputTensor;
};

#endif

