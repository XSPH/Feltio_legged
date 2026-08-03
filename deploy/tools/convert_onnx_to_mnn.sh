#!/bin/bash
set -e  # 遇到错误立即退出

MNN_BUILD_DIR="/home/asuka/MNN/build"
ONNX_DIR="/home/asuka/Legged/Feltio_legged/logs/rough_go2/exported/policies"
MNN_OUT_DIR="/home/asuka/Legged/Feltio_legged/deploy/models"

mkdir -p "$MNN_OUT_DIR"

# 进入 MNN build 目录
if [ -d "$MNN_BUILD_DIR" ]; then
    cd "$MNN_BUILD_DIR"
    echo "Changed directory to: $(pwd)"
else
    echo "Error: MNN build directory not found at $MNN_BUILD_DIR"
    exit 1
fi

echo "------------------------------------------------"
echo "Converting policy model"
./MNNConvert \
    -f ONNX \
    --modelFile "$ONNX_DIR/policy.onnx" \
    --MNNModel "$MNN_OUT_DIR/policy.mnn"

echo "------------------------------------------------"
echo "All conversions completed successfully!"