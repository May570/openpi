import pandas as pd

# 设置数据文件路径
data_file = "/share/project/wujiling/datasets/tasks/pour_coffee_3924/data/chunk-000/episode_000000.parquet"  # 修改为您的实际文件路径

# 读取 .parquet 文件
df = pd.read_parquet(data_file)

# 打印数据框的前几行以查看数据结构
print("数据框的前几行：")
print(df.head())  # 显示前五行
print("数据框的第一个：")
print(df.iloc[0])  # 显示第一个
print("数据框的第2个：")
print(df.iloc[1])  # 显示第一个

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

if 'action' in df.columns:
    print("\n'action' 列的前几行：")
    print(df['action'].head())

if 'observation.state' in df.columns:
    first_state = df['observation.state'].iloc[0]
    print("第一个 state：")
    print(first_state)

if 'observation.images.cam_high' in df.columns:
    print("\n'observation.images.cam_high' 列的前几行：")
    print(df['observation.images.cam_high'].head())

if 'state' in df.columns:
    print("\n'state' 列的前几行：")
    print(df['state'].head())
    first_state = df['state'].iloc[0]
    print("第一个 state：")
    print(first_state)

if 'actions' in df.columns:
    print("\n'actions' 列的前几行：")
    print(df['actions'].head())
    first_action = df['actions'].iloc[0]
    print("第一个 action：")
    print(first_action)

if 'length' in df.columns:
    print("\n'length' 列的前几行：")
    print(df['length'])

if 'observation.images.top' in df.columns:
    print("\n'observation.images.top' 列的前几行：")
    print(df['observation.images.top'].head())