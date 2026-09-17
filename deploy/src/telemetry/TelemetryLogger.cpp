#include "telemetry/TelemetryLogger.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr std::array<const char*, 4> kFootNames = {
    "FL_foot", "FR_foot", "RL_foot", "RR_foot",
};

enum class Direction { UNKNOWN, UP, DOWN };

const char *directionName(Direction direction) {
    switch (direction) {
    case Direction::UP:
        return "up";
    case Direction::DOWN:
        return "down";
    default:
        return "unknown";
    }
}

const char *stateName(FSMStateName state) {
    switch (state) {
    case FSMStateName::PASSIVE:
        return "passive";
    case FSMStateName::FIXEDSTAND:
        return "fixed_stand";
    case FSMStateName::Rl:
        return "rl";
    default:
        return "invalid";
    }
}

Direction directionFromName(const std::string& name) {
    if (name.find("_up_") != std::string::npos ||
        (name.size() >= 3 && name.compare(name.size() - 3, 3, "_up") == 0)) {
        return Direction::UP;
    }
    if (name.find("_down_") != std::string::npos ||
        (name.size() >= 5 && name.compare(name.size() - 5, 5, "_down") == 0)) {
        return Direction::DOWN;
    }
    return Direction::UNKNOWN;
}

std::string csvQuote(const std::string& value) {
    if (value.find_first_of(",\"\n") == std::string::npos) {
        return value;
    }
    std::string quoted = "\"";
    for (char character : value) {
        if (character == '"') {
            quoted += "\"\"";
        }
        else {
            quoted += character;
        }
    }
    quoted += '"';
    return quoted;
}

std::string join(const std::vector<std::string>& values) {
    std::ostringstream result;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            result << ';';
        }
        result << values[i];
    }
    return result.str();
}

std::string timestamp() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t time = std::chrono::system_clock::to_time_t(now);
    std::tm local{};
    localtime_r(&time, &local);
    std::ostringstream result;
    result << std::put_time(&local, "%Y%m%d_%H%M%S");
    return result.str();
}

double percentile(std::vector<double> values, double probability) {
    if (values.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    std::sort(values.begin(), values.end());
    const double position = probability * static_cast<double>(values.size() - 1);
    const std::size_t lower = static_cast<std::size_t>(std::floor(position));
    const std::size_t upper = static_cast<std::size_t>(std::ceil(position));
    const double fraction = position - static_cast<double>(lower);
    return values[lower] * (1.0 - fraction) + values[upper] * fraction;
}

struct Metrics {
    int nosingContacts = 0;
    std::vector<double> impacts;
    std::vector<double> noSupportDurations;
    std::vector<double> stopRecoveries;
    int unrecoveredStops = 0;
};

struct FootSample {
    std::array<mjtNum, 3> position{};
    std::array<mjtNum, 3> velocity{};
    std::array<mjtNum, 3> force{};
    std::vector<std::string> contacts;
    bool contact = false;
    bool support = false;
    bool touchdown = false;
    std::array<mjtNum, 3> precontactVelocity{};
};

struct FootState {
    bool initialized = false;
    bool inSupport = false;
    double lastSupportTime = 0.0;
    std::array<mjtNum, 3> lastAirVelocity{};
};

struct NosingState {
    int robotGeom = -1;
    int terrainGeom = -1;
    Direction direction = Direction::UNKNOWN;
    double startTime = 0.0;
    double lastSeenTime = 0.0;
    double peakMagnitude = 0.0;
    std::array<mjtNum, 3> peakForce{};
    FSMStateName fsmState = FSMStateName::INVALID;
};

struct NoSupportState {
    bool active = false;
    double startTime = 0.0;
    Direction direction = Direction::UNKNOWN;
    FSMStateName fsmState = FSMStateName::INVALID;
};

struct StopState {
    bool active = false;
    double startTime = 0.0;
    Direction direction = Direction::UNKNOWN;
    FSMStateName fsmState = FSMStateName::INVALID;
};

std::string geomName(const mjModel *model, int geomId) {
    const char *name = mj_id2name(model, mjOBJ_GEOM, geomId);
    return name == nullptr ? "geom_" + std::to_string(geomId) : name;
}

void addUnique(std::vector<std::string>& values, const std::string& value) {
    if (std::find(values.begin(), values.end(), value) == values.end()) {
        values.push_back(value);
    }
}

}  // namespace

struct TelemetryLogger::Impl {
    Impl(const mjModel *modelValue, const mjData *dataValue,
         const DeployConfig& configValue)
        : model(modelValue), data(dataValue), config(configValue),
          enabled(configValue.logging.enabled) {
        if (!enabled) {
            return;
        }
        for (std::size_t foot = 0; foot < footGeomIds.size(); ++foot) {
            footGeomIds[foot] = mj_name2id(model, mjOBJ_GEOM, kFootNames[foot]);
            if (footGeomIds[foot] < 0) {
                throw std::runtime_error(std::string("Telemetry foot geom not found: ") +
                                         kFootNames[foot]);
            }
        }
        int rootJoint = mj_name2id(model, mjOBJ_JOINT, "root");
        if (rootJoint < 0) {
            for (int joint = 0; joint < model->njnt; ++joint) {
                if (model->jnt_type[joint] == mjJNT_FREE) {
                    rootJoint = joint;
                    break;
                }
            }
        }
        if (rootJoint < 0) {
            throw std::runtime_error("Telemetry free root joint not found");
        }
        rootQpos = model->jnt_qposadr[rootJoint];
        rootBody = model->jnt_bodyid[rootJoint];
        samplePeriod = 1.0 / config.logging.frequency_hz;
        createOutputFiles();
        writeHeaders();
    }
    ~Impl() {
        try {
            finalize();
        }
        catch (...) {
        }
    }

    void createOutputFiles() {
        std::filesystem::create_directories(config.logging.output_dir);
        outputDirectory = config.logging.output_dir / timestamp();
        int suffix = 1;
        while (std::filesystem::exists(outputDirectory)) {
            outputDirectory = config.logging.output_dir /
                (timestamp() + "_" + std::to_string(suffix++));
        }
        std::filesystem::create_directories(outputDirectory);
        frames.open(outputDirectory / "frames.csv");
        events.open(outputDirectory / "events.csv");
        if (!frames || !events) {
            throw std::runtime_error("Failed to open telemetry logs in " +
                                     outputDirectory.string());
        }
        std::cout << "[logging] " << outputDirectory << " ("
                  << config.logging.frequency_hz << " Hz)\n";
    }

    void writeHeaders() {
        frames << "session_time_s,episode_id,episode_time_s,sample_id,fsm_state,"
                  "target_vx,target_vy,target_wz,applied_vx,applied_vy,applied_wz,"
                  "base_x,base_y,base_z,base_qw,base_qx,base_qy,base_qz,"
                  "base_roll,base_pitch,base_yaw,base_vx,base_vy,base_vz,"
                  "base_wx,base_wy,base_wz,direction";
        for (const char *foot : kFootNames) {
            frames << ',' << foot << "_px," << foot << "_py," << foot << "_pz,"
                   << foot << "_vx," << foot << "_vy," << foot << "_vz,"
                   << foot << "_fx," << foot << "_fy," << foot << "_fz,"
                   << foot << "_contact," << foot << "_support,"
                   << foot << "_touchdown," << foot << "_pre_vx,"
                   << foot << "_pre_vy," << foot << "_pre_vz,"
                   << foot << "_contact_geoms";
        }
        frames << '\n';
        events << "event_id,episode_id,type,fsm_state,direction,start_time_s,end_time_s,"
                  "duration_s,foot,robot_geom,terrain_geom,pre_vx,pre_vy,pre_vz,"
                  "impact_speed,peak_fx,peak_fy,peak_fz,peak_force,status\n";
    }

    void sample(const CtrlComponents& ctrl, FSMStateName state) {
        if (!enabled || finalized) {
            return;
        }
        sessionTime += config.simulation_timestep;
        const double time = data->time;
        if (time + 1.0e-9 < nextSampleTime) {
            return;
        }
        do {
            nextSampleTime += samplePeriod;
        } while (nextSampleTime <= time + 1.0e-9);
        currentFsmState = state;

        std::array<FootSample, 4> feet{};
        for (std::size_t foot = 0; foot < feet.size(); ++foot) {
            mju_copy3(feet[foot].position.data(), data->geom_xpos + 3 * footGeomIds[foot]);
            mjtNum velocity[6]{};
            mj_objectVelocity(model, data, mjOBJ_GEOM, footGeomIds[foot], velocity, 0);
            mju_copy3(feet[foot].velocity.data(), velocity + 3);
        }

        std::unordered_map<long long, NosingState> seenNosings;
        Direction observedDirection = Direction::UNKNOWN;
        for (int contactId = 0; contactId < data->ncon; ++contactId) {
            const mjContact& contact = data->contact[contactId];
            if (contact.efc_address < 0) {
                continue;
            }
            mjtNum localForce[6]{};
            mj_contactForce(model, data, contactId, localForce);
            mjtNum worldForce[3]{};
            mju_mulMatTVec(worldForce, contact.frame, localForce, 3, 3);
            const double magnitude = mju_norm3(worldForce);
            if (magnitude <= config.logging.contact_force_threshold_n) {
                continue;
            }
            for (std::size_t foot = 0; foot < feet.size(); ++foot) {
                int otherGeom = -1;
                double sign = 0.0;
                if (contact.geom[0] == footGeomIds[foot]) {
                    otherGeom = contact.geom[1];
                    sign = -1.0;
                }
                else if (contact.geom[1] == footGeomIds[foot]) {
                    otherGeom = contact.geom[0];
                    sign = 1.0;
                }
                if (otherGeom >= 0) {
                    for (int axis = 0; axis < 3; ++axis) {
                        feet[foot].force[axis] += sign * worldForce[axis];
                    }
                    feet[foot].contact = true;
                    const std::string otherName = geomName(model, otherGeom);
                    addUnique(feet[foot].contacts, otherName);
                    const Direction direction = directionFromName(otherName);
                    if (direction != Direction::UNKNOWN) {
                        observedDirection = direction;
                    }
                }
            }

            const bool firstRobot = model->geom_bodyid[contact.geom[0]] != 0;
            const bool secondRobot = model->geom_bodyid[contact.geom[1]] != 0;
            if (firstRobot == secondRobot) {
                continue;
            }
            const int robotGeom = firstRobot ? contact.geom[0] : contact.geom[1];
            const int terrainGeom = firstRobot ? contact.geom[1] : contact.geom[0];
            const std::string terrainName = geomName(model, terrainGeom);
            if (terrainName.find("stair_nosing_") == std::string::npos) {
                continue;
            }
            const long long key = (static_cast<long long>(robotGeom) << 32) |
                                  static_cast<unsigned int>(terrainGeom);
            NosingState& seen = seenNosings[key];
            seen.robotGeom = robotGeom;
            seen.terrainGeom = terrainGeom;
            seen.direction = directionFromName(terrainName);
            seen.fsmState = currentFsmState;
            std::array<mjtNum, 3> robotForce{};
            const double sign = firstRobot ? -1.0 : 1.0;
            for (int axis = 0; axis < 3; ++axis) {
                robotForce[axis] = sign * worldForce[axis];
            }
            if (magnitude > seen.peakMagnitude) {
                seen.peakMagnitude = magnitude;
                seen.peakForce = robotForce;
            }
        }

        if (observedDirection != Direction::UNKNOWN) {
            lastDirection = observedDirection;
        }
        const Direction frameDirection = observedDirection == Direction::UNKNOWN
                                       ? lastDirection : observedDirection;
        updateNosings(seenNosings, time);
        updateFeet(feet, time, frameDirection);
        updateNoSupport(feet, time, frameDirection);
        updateStop(ctrl, feet, time, frameDirection);
        writeFrame(ctrl, state, feet, time, frameDirection);
        ++sampleId;
        if (sessionTime + 1.0e-9 >= nextFlushTime) {
            frames.flush();
            events.flush();
            nextFlushTime = sessionTime + config.logging.flush_interval_s;
        }
    }

    void updateNosings(const std::unordered_map<long long, NosingState>& seen,
                       double time) {
        for (const auto& item : seen) {
            auto active = activeNosings.find(item.first);
            if (active == activeNosings.end()) {
                NosingState state = item.second;
                state.startTime = time;
                state.lastSeenTime = time;
                activeNosings.emplace(item.first, state);
            }
            else {
                active->second.lastSeenTime = time;
                if (item.second.peakMagnitude > active->second.peakMagnitude) {
                    active->second.peakMagnitude = item.second.peakMagnitude;
                    active->second.peakForce = item.second.peakForce;
                }
            }
        }
        for (auto active = activeNosings.begin(); active != activeNosings.end();) {
            if (seen.find(active->first) == seen.end() &&
                time - active->second.lastSeenTime >
                    config.logging.contact_gap_tolerance_s + 1.0e-9) {
                finishNosing(active->second);
                active = activeNosings.erase(active);
            }
            else {
                ++active;
            }
        }
    }

    void finishNosing(const NosingState& state) {
        const std::string robotName = geomName(model, state.robotGeom);
        const std::string terrainName = geomName(model, state.terrainGeom);
        writeEvent("nosing_contact", state.direction, state.startTime,
                   state.lastSeenTime, "", robotName, terrainName, {}, 0.0,
                   state.peakForce, state.peakMagnitude, "complete", state.fsmState);
        addMetric(state.direction, state.fsmState,
                  [](Metrics& metrics) { ++metrics.nosingContacts; });
    }

    void updateFeet(std::array<FootSample, 4>& feet, double time,
                    Direction direction) {
        for (std::size_t foot = 0; foot < feet.size(); ++foot) {
            FootState& state = footStates[foot];
            feet[foot].support = feet[foot].force[2] >
                                 config.logging.support_force_threshold_n;
            if (!state.initialized) {
                state.initialized = true;
                state.inSupport = feet[foot].support;
                state.lastSupportTime = time;
                state.lastAirVelocity = feet[foot].velocity;
                continue;
            }
            if (!feet[foot].support) {
                state.lastAirVelocity = feet[foot].velocity;
                if (state.inSupport && time - state.lastSupportTime >
                    config.logging.contact_gap_tolerance_s + 1.0e-9) {
                    state.inSupport = false;
                }
                continue;
            }
            state.lastSupportTime = time;
            if (!state.inSupport) {
                state.inSupport = true;
                feet[foot].touchdown = true;
                feet[foot].precontactVelocity = state.lastAirVelocity;
                const double impact = std::max(0.0,
                    -static_cast<double>(state.lastAirVelocity[2]));
                const std::string terrain = join(feet[foot].contacts);
                writeEvent("touchdown", direction, time, time, kFootNames[foot],
                           kFootNames[foot], terrain, state.lastAirVelocity, impact,
                           feet[foot].force, mju_norm3(feet[foot].force.data()),
                           "complete", currentFsmState);
                addMetric(direction, currentFsmState, [impact](Metrics& metrics) {
                    metrics.impacts.push_back(impact);
                });
            }
        }
    }

    void updateNoSupport(const std::array<FootSample, 4>& feet, double time,
                         Direction direction) {
        const bool anySupport = std::any_of(feet.begin(), feet.end(),
            [](const FootSample& foot) { return foot.support; });
        if (!anySupport && !noSupport.active) {
            noSupport.active = true;
            noSupport.startTime = time;
            noSupport.direction = direction;
            noSupport.fsmState = currentFsmState;
        }
        else if (anySupport && noSupport.active) {
            if (direction != Direction::UNKNOWN) {
                noSupport.direction = direction;
            }
            finishNoSupport(time);
        }
    }

    void finishNoSupport(double time) {
        const double duration = std::max(0.0, time - noSupport.startTime);
        writeEvent("no_support", noSupport.direction, noSupport.startTime, time,
                   "", "", "", {}, 0.0, {}, 0.0, "complete",
                   noSupport.fsmState);
        addMetric(noSupport.direction, noSupport.fsmState,
                  [duration](Metrics& metrics) {
            metrics.noSupportDurations.push_back(duration);
        });
        noSupport = {};
    }

    void updateStop(const CtrlComponents& ctrl,
                    const std::array<FootSample, 4>& feet, double time,
                    Direction direction) {
        const auto& target = ctrl.targetVelocityCommand;
        const bool commandActive = std::hypot(target[0], target[1]) >
                                   config.logging.stop_linear_threshold ||
                                   std::abs(target[2]) >
                                   config.logging.stop_angular_threshold;
        if (previousCommandActive && !commandActive) {
            stop.active = true;
            stop.startTime = time;
            stop.direction = direction;
            stop.fsmState = currentFsmState;
        }
        else if (!previousCommandActive && commandActive && stop.active) {
            writeEvent("stop_recovery", stop.direction, stop.startTime, time,
                       "", "", "", {}, 0.0, {}, 0.0, "cancelled",
                       stop.fsmState);
            stop = {};
        }
        previousCommandActive = commandActive;

        const bool allSupport = std::all_of(feet.begin(), feet.end(),
            [](const FootSample& foot) { return foot.support; });
        if (allSupport) {
            if (!fourFootSupportActive) {
                fourFootSupportActive = true;
                fourFootSupportStart = time;
            }
        }
        else {
            fourFootSupportActive = false;
        }
        if (stop.active && fourFootSupportActive &&
            time - fourFootSupportStart + 1.0e-9 >=
                config.logging.four_foot_stable_s) {
            const bool stableBeforeStop = fourFootSupportStart <=
                stop.startTime - config.logging.four_foot_stable_s + 1.0e-9;
            const double endTime = stableBeforeStop ? stop.startTime : time;
            const double recovery = endTime - stop.startTime;
            writeEvent("stop_recovery", stop.direction, stop.startTime, endTime,
                       "", "", "", {}, 0.0, {}, 0.0, "recovered",
                       stop.fsmState);
            addMetric(stop.direction, stop.fsmState, [recovery](Metrics& metrics) {
                metrics.stopRecoveries.push_back(recovery);
            });
            stop = {};
        }
    }

    template <typename Function>
    void addMetric(Direction direction, FSMStateName state, Function function) {
        if (state != FSMStateName::Rl) {
            return;
        }
        function(episodeMetrics[direction]);
        function(episodeMetricsAll);
    }

    void writeEvent(const std::string& type, Direction direction,
                    double startTime, double endTime, const std::string& foot,
                    const std::string& robotGeom, const std::string& terrainGeom,
                    const std::array<mjtNum, 3>& preVelocity, double impact,
                    const std::array<mjtNum, 3>& peakForce, double peakMagnitude,
                    const std::string& status, FSMStateName state) {
        events << eventId++ << ',' << episodeId << ',' << type << ','
               << stateName(state) << ',' << directionName(direction) << ',' << std::fixed
               << std::setprecision(6) << startTime << ',' << endTime << ','
               << std::max(0.0, endTime - startTime) << ',' << csvQuote(foot) << ','
               << csvQuote(robotGeom) << ',' << csvQuote(terrainGeom) << ','
               << preVelocity[0] << ',' << preVelocity[1] << ',' << preVelocity[2]
               << ',' << impact << ',' << peakForce[0] << ',' << peakForce[1]
               << ',' << peakForce[2] << ',' << peakMagnitude << ',' << status
               << '\n';
    }

    void writeFrame(const CtrlComponents& ctrl, FSMStateName state,
                    const std::array<FootSample, 4>& feet, double time,
                    Direction direction) {
        const mjtNum *qpos = data->qpos + rootQpos;
        const double w = qpos[3];
        const double x = qpos[4];
        const double y = qpos[5];
        const double z = qpos[6];
        const double roll = std::atan2(2.0 * (w*x + y*z),
                                       1.0 - 2.0 * (x*x + y*y));
        const double pitchValue = std::clamp(2.0 * (w*y - z*x), -1.0, 1.0);
        const double pitch = std::asin(pitchValue);
        const double yaw = std::atan2(2.0 * (w*z + x*y),
                                      1.0 - 2.0 * (y*y + z*z));
        mjtNum baseVelocity[6]{};
        mj_objectVelocity(model, data, mjOBJ_BODY, rootBody, baseVelocity, 0);
        frames << std::fixed << std::setprecision(6) << sessionTime << ','
               << episodeId << ',' << time << ',' << sampleId << ','
               << stateName(state);
        for (float value : ctrl.targetVelocityCommand) {
            frames << ',' << value;
        }
        for (float value : ctrl.appliedVelocityCommand) {
            frames << ',' << value;
        }
        frames << ',' << qpos[0] << ',' << qpos[1] << ',' << qpos[2]
               << ',' << w << ',' << x << ',' << y << ',' << z
               << ',' << roll << ',' << pitch << ',' << yaw
               << ',' << baseVelocity[3] << ',' << baseVelocity[4]
               << ',' << baseVelocity[5] << ',' << baseVelocity[0]
               << ',' << baseVelocity[1] << ',' << baseVelocity[2]
               << ',' << directionName(direction);
        for (const FootSample& foot : feet) {
            frames << ',' << foot.position[0] << ',' << foot.position[1]
                   << ',' << foot.position[2] << ',' << foot.velocity[0]
                   << ',' << foot.velocity[1] << ',' << foot.velocity[2]
                   << ',' << foot.force[0] << ',' << foot.force[1]
                   << ',' << foot.force[2] << ',' << foot.contact
                   << ',' << foot.support << ',' << foot.touchdown
                   << ',' << foot.precontactVelocity[0]
                   << ',' << foot.precontactVelocity[1]
                   << ',' << foot.precontactVelocity[2]
                   << ',' << csvQuote(join(foot.contacts));
        }
        frames << '\n';
    }

    void resetEpisode() {
        if (!enabled || finalized) {
            return;
        }
        closeEpisode(data->time);
        ++episodeId;
        nextSampleTime = 0.0;
        lastDirection = Direction::UNKNOWN;
        footStates = {};
        previousCommandActive = false;
        fourFootSupportActive = false;
        fourFootSupportStart = 0.0;
    }

    void closeEpisode(double time) {
        for (const auto& active : activeNosings) {
            finishNosing(active.second);
        }
        activeNosings.clear();
        if (noSupport.active) {
            finishNoSupport(time);
        }
        if (stop.active) {
            writeEvent("stop_recovery", stop.direction, stop.startTime, time,
                       "", "", "", {}, 0.0, {}, 0.0, "unrecovered",
                       stop.fsmState);
            addMetric(stop.direction, stop.fsmState, [](Metrics& metrics) {
                ++metrics.unrecoveredStops;
            });
            stop = {};
        }
        episodeResults.push_back({episodeId, episodeMetrics, episodeMetricsAll});
        episodeMetrics.clear();
        episodeMetricsAll = {};
    }

    void finalize() {
        if (!enabled || finalized) {
            return;
        }
        closeEpisode(data->time);
        writeSummary();
        frames.flush();
        events.flush();
        frames.close();
        events.close();
        finalized = true;
    }

    void writeSummary() {
        std::ofstream summary(outputDirectory / "summary.csv");
        if (!summary) {
            throw std::runtime_error("Failed to open telemetry summary");
        }
        summary << "scope,episode_id,direction,nosing_contacts,impact_count,"
                   "impact_p50,impact_p90,impact_p95,impact_p99,impact_max,"
                   "no_support_count,no_support_total_s,no_support_p50_s,"
                   "no_support_p95_s,no_support_max_s,stop_recovery_count,"
                   "stop_recovery_p50_s,stop_recovery_p95_s,"
                   "stop_recovery_max_s,unrecovered_stops\n";
        Metrics globalAll;
        std::map<Direction, Metrics> global;
        for (const EpisodeResult& result : episodeResults) {
            writeSummaryRow(summary, "episode", result.id, "all", result.all);
            mergeMetrics(globalAll, result.all);
            for (Direction direction : {Direction::UP, Direction::DOWN,
                                        Direction::UNKNOWN}) {
                const auto metrics = result.byDirection.find(direction);
                const Metrics empty;
                writeSummaryRow(summary, "episode", result.id,
                                directionName(direction),
                                metrics == result.byDirection.end()
                                    ? empty : metrics->second);
                if (metrics != result.byDirection.end()) {
                    mergeMetrics(global[direction], metrics->second);
                }
            }
        }
        writeSummaryRow(summary, "global", -1, "all", globalAll);
        for (Direction direction : {Direction::UP, Direction::DOWN,
                                    Direction::UNKNOWN}) {
            writeSummaryRow(summary, "global", -1, directionName(direction),
                            global[direction]);
        }
    }

    static void mergeMetrics(Metrics& destination, const Metrics& source) {
        destination.nosingContacts += source.nosingContacts;
        destination.impacts.insert(destination.impacts.end(), source.impacts.begin(),
                                   source.impacts.end());
        destination.noSupportDurations.insert(destination.noSupportDurations.end(),
            source.noSupportDurations.begin(), source.noSupportDurations.end());
        destination.stopRecoveries.insert(destination.stopRecoveries.end(),
            source.stopRecoveries.begin(), source.stopRecoveries.end());
        destination.unrecoveredStops += source.unrecoveredStops;
    }

    static void writeValue(std::ostream& output, double value) {
        if (std::isfinite(value)) {
            output << value;
        }
    }

    static void writeSummaryRow(std::ostream& output, const char *scope,
                                int episode, const char *direction,
                                const Metrics& metrics) {
        const double impactMax = metrics.impacts.empty() ?
            std::numeric_limits<double>::quiet_NaN() :
            *std::max_element(metrics.impacts.begin(), metrics.impacts.end());
        const double noSupportTotal = std::accumulate(
            metrics.noSupportDurations.begin(), metrics.noSupportDurations.end(), 0.0);
        const double noSupportMax = metrics.noSupportDurations.empty() ?
            std::numeric_limits<double>::quiet_NaN() : *std::max_element(
                metrics.noSupportDurations.begin(), metrics.noSupportDurations.end());
        const double recoveryMax = metrics.stopRecoveries.empty() ?
            std::numeric_limits<double>::quiet_NaN() : *std::max_element(
                metrics.stopRecoveries.begin(), metrics.stopRecoveries.end());
        output << scope << ',' << episode << ',' << direction << ','
               << metrics.nosingContacts << ',' << metrics.impacts.size() << ',';
        writeValue(output, percentile(metrics.impacts, 0.50)); output << ',';
        writeValue(output, percentile(metrics.impacts, 0.90)); output << ',';
        writeValue(output, percentile(metrics.impacts, 0.95)); output << ',';
        writeValue(output, percentile(metrics.impacts, 0.99)); output << ',';
        writeValue(output, impactMax); output << ','
            << metrics.noSupportDurations.size() << ',' << noSupportTotal << ',';
        writeValue(output, percentile(metrics.noSupportDurations, 0.50)); output << ',';
        writeValue(output, percentile(metrics.noSupportDurations, 0.95)); output << ',';
        writeValue(output, noSupportMax); output << ','
            << metrics.stopRecoveries.size() << ',';
        writeValue(output, percentile(metrics.stopRecoveries, 0.50)); output << ',';
        writeValue(output, percentile(metrics.stopRecoveries, 0.95)); output << ',';
        writeValue(output, recoveryMax); output << ','
            << metrics.unrecoveredStops << '\n';
    }

    struct EpisodeResult {
        int id;
        std::map<Direction, Metrics> byDirection;
        Metrics all;
    };

    const mjModel *model;
    const mjData *data;
    const DeployConfig& config;
    bool enabled = false;
    bool finalized = false;
    std::filesystem::path outputDirectory;
    std::ofstream frames;
    std::ofstream events;
    std::array<int, 4> footGeomIds{};
    std::array<FootState, 4> footStates{};
    int rootQpos = -1;
    int rootBody = -1;
    int episodeId = 0;
    long long sampleId = 0;
    long long eventId = 0;
    double samplePeriod = 0.005;
    double nextSampleTime = 0.0;
    double sessionTime = 0.0;
    double nextFlushTime = 0.0;
    Direction lastDirection = Direction::UNKNOWN;
    std::unordered_map<long long, NosingState> activeNosings;
    NoSupportState noSupport;
    StopState stop;
    bool previousCommandActive = false;
    bool fourFootSupportActive = false;
    double fourFootSupportStart = 0.0;
    FSMStateName currentFsmState = FSMStateName::INVALID;
    std::map<Direction, Metrics> episodeMetrics;
    Metrics episodeMetricsAll;
    std::vector<EpisodeResult> episodeResults;
};

TelemetryLogger::TelemetryLogger(const mjModel *model, const mjData *data,
                                 const DeployConfig& config)
    : impl_(std::make_unique<Impl>(model, data, config)){}

TelemetryLogger::~TelemetryLogger() = default;

void TelemetryLogger::sample(const CtrlComponents& ctrl, FSMStateName state) {
    impl_->sample(ctrl, state);
}

void TelemetryLogger::resetEpisode() {
    impl_->resetEpisode();
}

void TelemetryLogger::finalize() {
    impl_->finalize();
}
