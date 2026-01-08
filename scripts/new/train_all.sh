#!/bin/bash
set -e

# 日志目录
LOG_DIR="train_logs"
mkdir -p $LOG_DIR

# 左边是模型 name，右边是 config（第一个参数）
declare -A MODELS=(
  ["pi05_agilex_pencil_sharpener"]="pi05_agilex_pencil_sharpener"
  ["pi05_agilex_bread"]="pi05_agilex_bread"
  ["pi05_agilex_pot"]="pi05_agilex_pot"
  ["pi05_agilex_fruit"]="pi05_agilex_fruit"
  ["pi05_agilex_nearest_toothpaste"]="pi05_agilex_nearest_toothpaste"
)

for NAME in "${!MODELS[@]}"; do
  CONFIG="${MODELS[$NAME]}"
  LOGFILE="${LOG_DIR}/${NAME}.log"

  echo "🚀 开始训练: $NAME"
  echo "日志保存到: $LOGFILE"

  python -u scripts/train.py "$CONFIG" \
    --exp-name="$NAME" \
    --batch_size=8 \
    --fsdp_devices=2 \
    --overwrite \
    2>&1 | tee -a "$LOGFILE"
done
