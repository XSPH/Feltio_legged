#!/bin/bash
set -e  # 遇到错误立即退出

MNN_CONVERTER="${MNN_CONVERTER:-/home/asuka/MNN/build/MNNConvert}"
ONNX_MODEL="${1:-/home/asuka/Legged/Feltio_legged/logs/rough_go2_tshim/exported/policies/policy.onnx}"
MNN_MODEL="${2:-/home/asuka/Legged/Feltio_legged/deploy/models/policy.mnn}"

mkdir -p "$(dirname "$MNN_MODEL")"

echo "------------------------------------------------"
echo "Converting policy model"
"$MNN_CONVERTER" \
    -f ONNX \
    --modelFile "$ONNX_MODEL" \
    --MNNModel "$MNN_MODEL"

echo "------------------------------------------------"
echo "All conversions completed successfully!"
