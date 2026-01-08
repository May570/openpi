import pandas as pd
import numpy as np
import cv2

# 读 parquet
df = pd.read_parquet("/share/project/wujiling/data/processed/fold_clothes/data/chunk-000/episode_000001.parquet")

# # 提取 action 列
# # 注意：action 列里每个元素可能是 list/array，需要转成 np.array
# actions = np.stack(df["action"].to_numpy())

# print("actions shape:", actions.shape)  # (T, action_dim)

# # 保存为 npy
# np.save("/home/admin123/桌面/actions_episode_000003.npy", actions)
# print("保存完成: actions_episode_000003.npy")

# === 提取 top 列，保存成视频 ===
frames = []
for row in df["observation.images.cam_high"]:
    img_bytes = row["bytes"]  # 取出字典里的 bytes
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)  # 解码为 BGR 图像
    frames.append(img)

h, w, c = frames[0].shape
out_path = "/share/project/wujiling/openpi/scripts/new/cam_high.mp4"

# 使用 MP4V 编码器
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = 50  # 可以改成实际帧率
out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

for f in frames:
    out.write(f)  # 已经是 BGR，不需要再转换

out.release()
print("视频已保存:", out_path)