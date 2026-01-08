#!/bin/bash
# A2D数据处理并行脚本
# 使用前需要修改TASK_NAME, START_TASK_ID, END_TASK_ID, TOTAL_TASKS

export SAMPLE_INTERVAL=1
export CHUNK=30

# ========== 配置区域 ==========
# 任务类型: pour (倒水), open_close (开关抽屉), erase (擦黑板), pnp(抓取放置), pour_coffee_105(倒咖啡)
TASK_NAME="pour_coffee_105_"

# raw_task 指令（可选，留空则自动从标注文件的 action_config 中提取并拼接）
# 如果标注文件中有多个子任务，会自动用 " and " 拼接
# 例如: "pick up the cup and place it" and "pour the coffee" -> "pick up the cup and place it and pour the coffee"
INSTRUCTION=""

# Task ID范围
# pour (倒水): 2218-2575
# open_close (开关抽屉): 2846-3112
# erase (擦黑板): 3149-3153
# pour_coffee_105 (倒咖啡): 2630-2632
START_TASK_ID=3924
END_TASK_ID=3928

# 总任务数（需要先运行count_trajectories_range.py统计）
TOTAL_TASKS=105  # ⚠️ 这个需要实际统计后填写

# 并行进程数
NUM_PROCESSES=53  # 根据TOTAL_TASKS调整
# ========== 配置区域结束 ==========

# 路径配置
DATA_DIR="/share/project/caomingyu/a2d_data"
OUTPUT_DIR="/share/project/fengli/data"
CODE_DIR="/share/project/fengli/code"
NORM_FILE="${CODE_DIR}/norm/a2d_norm_${TASK_NAME}.json"

# 日志目录
LOG_DIR="${OUTPUT_DIR}/${TASK_NAME}/logs"
mkdir -p $LOG_DIR

# 计算每个进程处理的任务数量
BASE_CHUNK=$((TOTAL_TASKS / NUM_PROCESSES))
EXTRA=$((TOTAL_TASKS % NUM_PROCESSES))

# 初始化任务记录
declare -a TASK_LOGS=()
declare -a PIDS=()

START_IDX=0
for ((i=0; i<NUM_PROCESSES; i++)); do
  CHUNK_SIZE=$BASE_CHUNK
  if [ $i -lt $EXTRA ]; then
    CHUNK_SIZE=$((CHUNK_SIZE + 1))
  fi

  END_IDX=$((START_IDX + CHUNK_SIZE))
  LOG_NAME="log_$i"

  # 构建命令参数
  CMD=(
    python3 "${CODE_DIR}/pretrain_data_process_a2d.py"
    --task_name "${TASK_NAME}"
    --data_dir "${DATA_DIR}"
    --root_path "${OUTPUT_DIR}"
    --normal_path "${NORM_FILE}"
    --start_task_id "${START_TASK_ID}"
    --end_task_id "${END_TASK_ID}"
    --start_idx "${START_IDX}"
    --end_idx "${END_IDX}"
    --chunk "${CHUNK}"
  )
  
  # 只有当 INSTRUCTION 不为空时才添加 --instruction 参数
  if [ -n "${INSTRUCTION}" ]; then
    CMD+=(--instruction "${INSTRUCTION}")
  fi

  echo "启动任务: $LOG_NAME（日志: $LOG_DIR/$LOG_NAME.log）"
  "${CMD[@]}" > "$LOG_DIR/$LOG_NAME.log" 2>&1 &
  pid=$!
  PIDS+=($pid)
  TASK_LOGS+=("$LOG_NAME")
  echo "任务PID: $pid" >> "$LOG_DIR/process_ids.log"

  START_IDX=$END_IDX
done

echo "=========================================="
echo "任务类型: ${TASK_NAME}"
echo "Task ID范围: ${START_TASK_ID} - ${END_TASK_ID}"
echo "总轨迹数: ${TOTAL_TASKS}"
echo "并行进程数: ${NUM_PROCESSES}"
echo "日志目录: ${LOG_DIR}"
echo "=========================================="
echo ""

# 等待所有并行任务完成
wait

# 检查所有任务是否成功
echo -e "\n========== 所有任务运行结束 =========="
for log_name in "${TASK_LOGS[@]}"; do
  # 检查是否有"Finish"或"处理完成"
  if grep -qi "Finish\|处理完成" "$LOG_DIR/$log_name.log"; then
    echo "✅ $log_name 运行成功"
  else
    echo "❌ $log_name 运行失败！查看日志: $LOG_DIR/$log_name.log"
  fi
done

echo -e "\n下一步: 运行merge_jsonl.py合并JSONL文件"
echo "python3 ${CODE_DIR}/merge_jsonl.py --task_name ${TASK_NAME} --data_dir ${OUTPUT_DIR}"
