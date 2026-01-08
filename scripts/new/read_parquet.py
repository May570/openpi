import os
import pandas as pd

# # 数据目录
# data_dir = "/share/project/section/task/orange_200_9_8/data/chunk-000"

# # 遍历目录下所有 parquet 文件
# for fname in sorted(os.listdir(data_dir)):
#     if fname.endswith(".parquet"):
#         file_path = os.path.join(data_dir, fname)
#         print(f"\n==== 文件: {fname} ====")

#         try:
#             df = pd.read_parquet(file_path)

#             if 'observation.state' in df.columns:
#                 states = df['observation.state']
#                 # 打印前两行 state
#                 print("前两个 state:")
#                 for i in range(min(2, len(states))):
#                     print(f"  state[{i}]: {states.iloc[i]}")
#             else:
#                 print("⚠️ 该文件没有 'observation.state' 列")

#         except Exception as e:
#             print(f"❌ 读取 {fname} 失败: {e}")

# 设置数据文件路径
data_file = "/share/project/wujiling/openpi/datasets/tasks/banana/data/chunk-000/episode_000000.parquet"

# 读取 .parquet 文件
df = pd.read_parquet(data_file)

# 打印数据框的前几行以查看数据结构
print("数据框的前几行：")
print(df.head())  # 显示前五行

# 打印数据框的列名，查看每一列的数据
print("\n数据框的列名：")
print(df.columns)

# 打印数据框每列的数据类型
print("\n每列的数据类型：")
print(df.dtypes)

# 检查某些列的具体内容（例如 'observation.state' 和 'observation.images.cam_high'）
if 'observation.state' in df.columns:
    print("\n'observation.state' 列的前几行：")
    print(df['observation.state'].head())

if 'observation.state' in df.columns:
    first_state = df['observation.state'].iloc[0]
    print("第一个 state：")
    print(first_state)

if 'observation.images.cam_high' in df.columns:
    print("\n'observation.images.cam_high' 列的前几行：")
    print(df['observation.images.cam_high'].head())

