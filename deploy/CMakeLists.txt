cmake_minimum_required(VERSION 3.16)
project(go2_mujoco_deploy LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Eigen3 REQUIRED)
find_package(glfw3 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Threads REQUIRED)

set(MUJOCO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/mujoco-3.2.5")
set(MNN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/MNN")

set(DEPLOY_SOURCES
    main.cpp
    src/config/DeployConfig.cpp
    src/FSM/FSM.cpp
    src/FSM/FSMState.cpp
    src/FSM/State_Passive.cpp
    src/FSM/State_FixedStand.cpp
    src/FSM/State_Rl.cpp
    src/control/ControlFrame.cpp
    src/control/rl_Inference.cpp
    src/interface/IOMujoco.cpp
    src/interface/WirelessHandle.cpp
)

add_executable(dogsim ${DEPLOY_SOURCES})

target_include_directories(dogsim PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${MUJOCO_ROOT}/include"
)

target_compile_options(dogsim PRIVATE -Wall -Wextra -Wpedantic)

target_link_libraries(dogsim PRIVATE
    Eigen3::Eigen
    glfw
    yaml-cpp
    Threads::Threads
    "${MUJOCO_ROOT}/lib/libmujoco.so"
    "${MNN_ROOT}/libMNN.so"
)

set_target_properties(dogsim PROPERTIES
    BUILD_RPATH "${MUJOCO_ROOT}/lib;${MNN_ROOT}"
)
