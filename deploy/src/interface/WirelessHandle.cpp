#include "interface/WirelessHandle.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <iostream>

#include <fcntl.h>
#include <linux/joystick.h>
#include <sys/select.h>
#include <unistd.h>

namespace {
constexpr unsigned char kButtonA = 0;
constexpr unsigned char kButtonB = 1;
constexpr unsigned char kButtonY = 3;
constexpr unsigned char kAxisLeftX = 0;
constexpr unsigned char kAxisLeftY = 1;
constexpr unsigned char kAxisRightX = 3;
constexpr unsigned char kAxisRightY = 4;
}  // namespace

WirelessHandle::WirelessHandle(std::string device)
    : _device(std::move(device)), _xboxFd(-1), _running(false) {
    _xboxFd = open(_device.c_str(), O_RDONLY | O_NONBLOCK);
    if (_xboxFd < 0) {
        std::cerr << "[joystick] " << _device << " unavailable: "
                  << std::strerror(errno)
                  << ". Keyboard P/F/R and movement controls remain available.\n";
        return;
    }
    _running.store(true);
    _readThread = std::thread(&WirelessHandle::readLoop, this);
    std::cout << "[joystick] using " << _device << '\n';
}

WirelessHandle::~WirelessHandle() {
    _running.store(false);
    if (_readThread.joinable()) {
        _readThread.join();
    }
    if (_xboxFd >= 0) {
        close(_xboxFd);
    }
}

float WirelessHandle::normalizeAxis(short value) {
    const float normalized = value >= 0 ? value / 32767.0F : value / 32768.0F;
    return std::clamp(normalized, -1.0F, 1.0F);
}

void WirelessHandle::readLoop() {
    while (_running.load()) {
        fd_set read_fds;
        FD_ZERO(&read_fds);
        FD_SET(_xboxFd, &read_fds);
        timeval timeout{0, 100000};
        const int ready = select(_xboxFd + 1, &read_fds, nullptr, nullptr, &timeout);
        if (ready <= 0) {
            continue;
        }

        js_event event{};
        const ssize_t count = read(_xboxFd, &event, sizeof(event));
        if (count != static_cast<ssize_t>(sizeof(event))) {
            continue;
        }
        const unsigned char type = event.type & ~JS_EVENT_INIT;
        if (type == JS_EVENT_BUTTON && event.value == 1) {
            if (event.number == kButtonA) {
                setUserCommand(UserCommand::RL);
            } else if (event.number == kButtonB) {
                setUserCommand(UserCommand::FIXED);
            } else if (event.number == kButtonY) {
                setUserCommand(UserCommand::PASS);
            }
        } else if (type == JS_EVENT_AXIS) {
            const float value = normalizeAxis(event.value);
            if (event.number == kAxisLeftX) {
                _axes.lx = value;
            } else if (event.number == kAxisLeftY) {
                _axes.ly = value;
            } else if (event.number == kAxisRightX) {
                _axes.rx = value;
            } else if (event.number == kAxisRightY) {
                _axes.ry = value;
            }
            setUserValue(_axes);
        }
    }
}
