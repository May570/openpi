#!/bin/bash
set -e

# ==== 配置 ====
ENV_PATH="/share/project/wujiling/env/pi05"
PYTHON_VERSION=3.11

# ==== 1. 创建 conda 环境 ====
echo "[1/6] Creating conda env at $ENV_PATH ..."
conda create --prefix "$ENV_PATH" python=$PYTHON_VERSION -y

# 激活环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate "$ENV_PATH"

# ==== 2. 基础工具 ====
echo "[2/6] Upgrade pip/setuptools/wheel ..."
pip install -U pip setuptools wheel

# ==== 3. 安装 openpi 主包 ====
echo "[3/6] Installing openpi (editable mode) ..."
pip install -e .

# ==== 4. 安装子包 ====
echo "[4/6] Installing sub-packages ..."
# openpi-client（本地 workspace）
pip install -e packages/openpi-client

# lerobot（固定 commit）
pip install git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5

# dlimp（固定 commit）
pip install git+https://github.com/kvablack/dlimp@ad72ce3a9b414db2185bc0b38461d4101a65477a

# ==== 5. 对齐 transformers 版本 ====
echo "[5/6] Installing transformers==4.53.2 ..."
pip install "transformers==4.53.2"

# # ==== 6. 打补丁 ====
# TRANSFORMERS_DIR=$(python -c "import transformers, pathlib; print(pathlib.Path(transformers.__file__).parent)")
# echo "[6/6] Copying transformers_replace/* into $TRANSFORMERS_DIR ..."
# cp -r ./src/openpi/models_pytorch/transformers_replace/* "$TRANSFORMERS_DIR"/

echo "✅ openpi_pytorch 环境安装完成！"
echo "要使用环境，请运行："
echo "    conda activate $ENV_PATH"
