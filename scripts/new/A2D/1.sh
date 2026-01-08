cd /share/project/fengli/code

# ========== 任务1: 倒水 (pour) ==========
# 注意：这里的范围是 job_id 范围（目录名），不是 task_id

# Step 1: 统计轨迹数（job_id 2218-2575）
python3 count_trajectories_range.py --start 2218 --end 2575
# 输出示例: 有效轨迹数: 250
# ✅ 记录这个数字

# Step 2: 修改 normal_a2d.py 第139行
#   TASK_NAME = "pour"
python3 normal_a2d.py
# ✅ 生成: /share/project/fengli/code/norm/a2d_norm_pour.json

# Step 3: 修改 pretrain_data_process_a2d.sh
#   TASK_NAME="pour"
#   INSTRUCTION=""  # 留空，自动从标注文件提取
#   START_TASK_ID=2218  # 这是 job_id 范围
#   END_TASK_ID=2575
#   TOTAL_TASKS=250       # 填Step 1的统计数字
#   NUM_PROCESSES=25      # 建议 TOTAL_TASKS/10
bash pretrain_data_process_a2d.sh
# ✅ 等待所有进程完成（查看日志监控进度）

# Step 4: 合并JSONL
python3 merge_jsonl.py --task_name pour
# ✅ 生成最终文件:
#   /share/project/fengli/data/pour/a2d_train_pour_320x240_merged.jsonl
#   /share/project/fengli/data/pour/a2d_train_pour_640x480_merged.jsonl



cd /share/project/fengli/code

# ========== 任务2: 开关抽屉 (open_close) ==========
# 注意：这里的范围是 job_id 范围（目录名），不是 task_id

# Step 1: 统计轨迹数（job_id 2846-3112）
python3 count_trajectories_range.py --start 2846 --end 3112
# ✅ 记录输出的有效轨迹数

# Step 2: 修改 normal_a2d.py 第139行
#   TASK_NAME = "open_close"
python3 normal_a2d.py
# ✅ 生成: /share/project/fengli/code/norm/a2d_norm_open_close.json

# Step 3: 修改 pretrain_data_process_a2d.sh
#   TASK_NAME="open_close"
#   INSTRUCTION=""  # 留空，自动从标注文件提取
#   START_TASK_ID=2846  # 这是 job_id 范围
#   END_TASK_ID=3112
#   TOTAL_TASKS=XXX       # 填Step 1的统计数字
#   NUM_PROCESSES=XX      # 建议 TOTAL_TASKS/10
bash pretrain_data_process_a2d.sh

# Step 4: 合并JSONL
python3 merge_jsonl.py --task_name open_close
# ✅ 生成最终文件:
#   /share/project/fengli/data/open_close/a2d_train_open_close_320x240_merged.jsonl
#   /share/project/fengli/data/open_close/a2d_train_open_close_640x480_merged.jsonl




cd /share/project/fengli/code

# ========== 任务3: 擦黑板 (erase) ==========
# 注意：这里的范围是 job_id 范围（目录名），不是 task_id

# Step 1: 统计轨迹数（job_id 3149-3153）
python3 count_trajectories_range.py --start 3149 --end 3153
# ✅ 记录输出的有效轨迹数

# Step 2: 修改 normal_a2d.py 第139行
#   TASK_NAME = "erase"
python3 normal_a2d.py
# ✅ 生成: /share/project/fengli/code/norm/a2d_norm_erase.json

# Step 3: 修改 pretrain_data_process_a2d.sh
#   TASK_NAME="erase"
#   INSTRUCTION=""  # 留空，自动从标注文件提取
#   START_TASK_ID=3149  # 这是 job_id 范围
#   END_TASK_ID=3153
#   TOTAL_TASKS=XXX       # 填Step 1的统计数字
#   NUM_PROCESSES=5       # 范围小，进程数可以少一些
bash pretrain_data_process_a2d.sh

# Step 4: 合并JSONL
python3 merge_jsonl.py --task_name erase
# ✅ 生成最终文件:
#   /share/project/fengli/data/erase/a2d_train_erase_320x240_merged.jsonl
#   /share/project/fengli/data/erase/a2d_train_erase_640x480_merged.jsonl