#!/usr/bin/env bash

# 遇到错误立即退出（可选，想稳一点就留着）
set -e

# 训练任务列表
TASKS=(
  pi05_blocks_ranking_rgb
  pi05_click_alarmclock
  pi05_move_can_pot
  pi05_place_bread_basket
  pi05_grab_roller
  pi05_handover_block
)

# 日志目录
LOG_DIR="train_logs"
mkdir -p "${LOG_DIR}"

# 逐个任务运行
for TASK in "${TASKS[@]}"; do
  echo "=============================================="
  echo "[INFO] Start training: ${TASK}"
  echo "=============================================="

  XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 \
  python -u scripts/train.py "${TASK}" \
    --exp-name="${TASK}" \
    --fsdp_devices=2 \
    --overwrite \
  2>&1 | tee -a "${LOG_DIR}/${TASK}.log"

  echo "[INFO] Finished: ${TASK}"
  echo
done

echo "✅ All tasks finished."
