#ifndef WIRELESS_HANDLE_H
#define WIRELESS_HANDLE_H

#include <atomic>
#include <string>
#include <thread>

#include "interface/CmdPanel.h"

class WirelessHandle final : public CmdPanel {
public:
    explicit WirelessHandle(std::string device);
    ~WirelessHandle() override;

    WirelessHandle(const WirelessHandle&) = delete;
    WirelessHandle& operator=(const WirelessHandle&) = delete;

private:
    void readLoop();
    static float normalizeAxis(short value);

    std::string _device;
    int _xboxFd;
    std::atomic<bool> _running;
    std::thread _readThread;
    UserValue _axes;
};

#endif
