# Go2 MuJoCo deployment

This directory contains only the MuJoCo simulation path and the three-state
controller (`passive -> fixed stand -> RL`). The RL state supports one
history-based student policy with 225 inputs and 12 actions. Each input contains
five 45-dimensional observations ordered from oldest to newest.

## Model

Export `policy.onnx` from `legged_gym/scripts/play.py`, then convert it:

```bash
MNN_CONVERTER=/path/to/MNNConvert \
  ./tools/convert_onnx_to_mnn.sh /path/to/policy.onnx models/policy.mnn
```

With no arguments, the script reads the policy exported under
`logs/rough_go2_tshim` and writes `deploy/models/policy.mnn`.
Convert a new model before running `dogsim`; an earlier 45-input
`policy.mnn` is rejected by the deployment shape check.

Set `model.path` in `configs/config.yaml` when selecting another policy.
`dogsim` always reads `deploy/configs/config.yaml`; no config command-line
argument is needed.

The exported model contains the student encoder and shared actor. The C++
controller keeps the five-frame history outside the model, clears it when
entering the RL state, and appends the current observation before each
inference step.

## Build and run

```bash
cmake -S . -B build
cmake --build build -j
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

The complete Go2 XML, URDF and mesh assets are copied unchanged from
`go2_rl_gym/resources/robots/go2`. The default scene is the original
`models/go2/flat.xml`.

The physics step is 0.005 s and the actor runs every four physics steps (50 Hz),
as configured in `configs/config.yaml`.
