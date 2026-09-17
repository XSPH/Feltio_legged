#ifndef TELEMETRYLOGGER_H
#define TELEMETRYLOGGER_H

#include <memory>

#include <mujoco/mujoco.h>

#include "common/enumClass.h"
#include "config/DeployConfig.h"
#include "control/CtrlComponents.h"

class TelemetryLogger {
public:
    TelemetryLogger(const mjModel *model, const mjData *data,
                    const DeployConfig& config);
    ~TelemetryLogger();

    TelemetryLogger(const TelemetryLogger&) = delete;
    TelemetryLogger& operator=(const TelemetryLogger&) = delete;

    void sample(const CtrlComponents& ctrl, FSMStateName state);
    void resetEpisode();
    void finalize();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

#endif
