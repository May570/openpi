#!/bin/bash
source /share/project/dumengfei/packages/miniconda3/bin/activate
conda activate lerobot

export FRAME_SAMPLE_INTERVAL=1
export ACTION_SAMPLE_INTERVAL=1
export CHUNK=30
export DATA_VERSION='real_data_a2d_0920_C30A1S1_S4_general_pnp'

# 日志目录（自动创建）
LOG_DIR="/share/project/dumengfei/code/pretrain_data_process/para_logs/${DATA_VERSION}"
mkdir -p $LOG_DIR  # 确保日志目录存在

# 总任务数与进程数
TOTAL_TASKS=583
NUM_PROCESSES=20

# 计算每个进程处理的任务数量
BASE_CHUNK=$((TOTAL_TASKS / NUM_PROCESSES))
EXTRA=$((TOTAL_TASKS % NUM_PROCESSES))  # 前 EXTRA 个任务多一个

# 初始化任务列表
declare -a TASKS=()

START_IDX=0
for ((i=0; i<NUM_PROCESSES; i++)); do
  # 分配额外任务
  CHUNK_SIZE=$BASE_CHUNK
  if [ $i -lt $EXTRA ]; then
    CHUNK_SIZE=$((CHUNK_SIZE + 1))
  fi

  END_IDX=$((START_IDX + CHUNK_SIZE))
  LOG_NAME="log_$i"
  TASKS+=("/share/project/dumengfei/code/pretrain_data_process/real_data/a2d/pretrain_data_process.py $LOG_NAME --task_start_idx $START_IDX --task_end_idx $END_IDX --chunk ${CHUNK}")
  START_IDX=$END_IDX
done

declare -a PIDS=()
# 并行启动所有任务
for task in "${TASKS[@]}"; do
  read py_file log_name args <<< "$task"

  # 后台运行并将输出重定向到日志文件
  echo "启动任务: $py_file $args（日志: $LOG_DIR/$log_name.log）"
  python3 $py_file $args > "$LOG_DIR/$log_name.log" 2>&1 &
  pid=$!
  PIDS+=($pid)
  echo "任务PID: $pid" >> "$LOG_DIR/process_ids.log"
done

# 等待所有并行任务完成
wait

# 检查所有任务是否成功
echo -e "\n===== 所有任务运行结束 ====="
for task in "${TASKS[@]}"; do
  read py_file log_name _ <<< "$task"
  if grep -qi "error\|exception" "$LOG_DIR/$log_name.log"; then
    echo "❌ $py_file 运行失败！查看日志: $LOG_DIR/$log_name.log"
  else
    echo "✅ $py_file 运行成功"
  fi
done

# python /share/project/dumengfei/code/pretrain_data_process/data_postprocess.py --data_name droid > "$LOG_DIR/postprocess.log" 2>&1 &
