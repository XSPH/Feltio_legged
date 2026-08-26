# Go2 and TOE_dog4.5 MuJoCo deployment

This directory contains only the MuJoCo simulation path and the three-state
controller (`passive -> fixed stand -> RL`). Its FSM and low-level command flow
follow the `toe_dog` deployment layout: each control tick receives state,
runs the active state, checks for a transition, and sends the cached motor
torques. The RL state supports one history-based student policy with 225 inputs
and 12 actions. Each input contains five 45-dimensional observations ordered
from oldest to newest.

## Model

Export `policy.onnx` from `legged_gym/scripts/play.py`, then convert it:

```bash
MNN_CONVERTER=/path/to/MNNConvert \
  ./tools/convert_onnx_to_mnn.sh /path/to/policy.onnx models/policy.mnn
```

With no arguments, the script reads `logs/exported/policy.onnx` and writes
`deploy/models/policy.mnn`.
Convert a new model before running `dogsim`; an earlier 45-input
`policy.mnn` is rejected by the deployment shape check.

Set `fel.model_path` in `configs/config.yaml` when selecting another policy.
`dogsim` always reads `deploy/configs/config.yaml`; no config command-line
argument is needed.

The exported model contains the student encoder and shared actor. The C++
controller keeps the five-frame history outside the model, clears it when
entering the RL state, and appends the current observation before each
inference step. The 45 values in each frame are ordered as angular velocity,
projected gravity, velocity command, joint-position error, joint velocity, and
last action. This fused model remains a single deployment artifact; it is not
split into the separate encoder and policy files used by `toe_dog`.

Policy, observation, action, gain, history, and command-ramp settings live in
the `fel` section. RL and fixed-stand gains are configured per joint type as
three-element `Kp` and `Kd` arrays.

## Build and run

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/dogsim
```

Controller mapping:

- Xbox `B` or keyboard `F`: fixed stand
- Xbox `A` or keyboard `R`: RL policy
- Xbox `Y` or keyboard `P`: passive
- Hold `W`/`S`: move forward/backward in RL mode
- Hold `A`/`D`: move left/right in RL mode
- Hold `Q`/`E`: turn left/right in RL mode
- Space: clear the keyboard movement command
- Backspace: reset simulation and FSM
- `Ctrl` + left-click and drag a robot link: apply an external force

Robot collision primitives are hidden in the viewer. This only changes their
display; collision detection and contact forces remain enabled. The floor and
terrain geometry remain visible.

Select the robot through `simulation.scene` in `configs/config.yaml`:

- Go2: `../models/go2/stairs.xml`
- TOE_dog4.5: `../../resources/robots/TOE_dog4.5/xml/scene.xml`

Joint order is shared by both robots. The runtime resolves the floating base
and each actuator from the loaded model, so switching scenes does not require a
C++ change. Select the policy model trained for the same robot through
`fel.model_path` at the same time.

The physics step is 0.005 s and the actor runs every four physics steps (50 Hz),
as configured in `configs/config.yaml`. Joint state and PD torque are refreshed
at every 200 Hz physics substep while the policy target is held for the full
policy period.
