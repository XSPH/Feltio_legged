#include "control/rl_Inference.h"

#include <algorithm>
#include <stdexcept>
#include <string>

#include "MNN/ErrorCode.hpp"

void checkTensor(MNN::Tensor *tensor, const char *name){
    if(tensor == nullptr){
        throw std::runtime_error(std::string("MNN ") + name + " tensor is null");
    }

    auto type = tensor->getType();
    if(type.code != halide_type_float || type.bits != 32){
        throw std::runtime_error(std::string("MNN ") + name + " tensor must be float32");
    }
}

rl_Inference::rl_Inference(const std::filesystem::path& modelPath, int numThreads)
    : _net(nullptr), _session(nullptr), _inputTensor(nullptr), _outputTensor(nullptr){
    _net = MNN::Interpreter::createFromFile(modelPath.c_str());
    if(_net == nullptr){
        throw std::runtime_error("Failed to load MNN model: " + modelPath.string());
    }

    MNN::ScheduleConfig scheduleConfig;
    scheduleConfig.type = MNN_FORWARD_CPU;
    scheduleConfig.numThread = numThreads;
    _session = _net->createSession(scheduleConfig);
    if(_session == nullptr){
        delete _net;
        _net = nullptr;
        throw std::runtime_error("Failed to create MNN session");
    }

    _inputTensor = _net->getSessionInput(_session, nullptr);
    _outputTensor = _net->getSessionOutput(_session, nullptr);
    checkTensor(_inputTensor, "input");
    checkTensor(_outputTensor, "output");

    if(_inputTensor->elementSize() != NUM_OBSERVATIONS){
        throw std::runtime_error("MNN input dimension must be 45");
    }
    if(_outputTensor->elementSize() != NUM_ACTIONS){
        throw std::runtime_error("MNN output dimension must be 12");
    }
}

rl_Inference::~rl_Inference(){
    if(_net != nullptr){
        if(_session != nullptr){
            _net->releaseSession(_session);
        }
        _net->releaseModel();
        delete _net;
    }
}

void rl_Inference::advanceNNsync(const float observation[NUM_OBSERVATIONS],
                                 float actionCmd[NUM_ACTIONS]){
    std::copy_n(observation, NUM_OBSERVATIONS, _inputTensor->host<float>());

    MNN::ErrorCode errorCode = _net->runSession(_session);
    if(errorCode != MNN::NO_ERROR){
        throw std::runtime_error("MNN inference failed with error code "
                                 + std::to_string(static_cast<int>(errorCode)));
    }

    std::copy_n(_outputTensor->host<float>(), NUM_ACTIONS, actionCmd);
}
