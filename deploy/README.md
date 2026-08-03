# Go2 MuJoCo deployment

This directory contains only the MuJoCo simulation path and the three-state
controller (`passive -> fixed stand -> RL`). The RL state supports one
feed-forward actor with 45 observations and 12 actions.

## Model

Export `policy.onnx` from `legged_gym/scripts/play.py`, then convert it:

```bash
MNN_CONVERTER=/path/to/MNNConvert \
  ./tools/convert_onnx_to_mnn.sh /path/to/policy.onnx models/policy.mnn
```

Set `model.path` in `configs/config.yaml` when selecting another policy.
`dogsim` always reads `deploy/configs/config.yaml`; no config command-line
argument is needed.

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
- Backspace: reset simulation and FSM

The complete Go2 XML, URDF and mesh assets are copied unchanged from
`go2_rl_gym/resources/robots/go2`. The default scene is the original
`models/go2/flat.xml`.

The physics step is 0.005 s and the actor runs every four physics steps (50 Hz),
as configured in `configs/config.yaml`.
