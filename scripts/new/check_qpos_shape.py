import os
import numpy as np
from pathlib import Path

# 指定轨迹目录
traj_dir = Path("/share/project/section/a2d_data/data/open_close/a2d_train_data/action_token/A2D0015AC00608_2846_97774")

def load_qpos_stack(traj_dir: Path, prefix: str):
    """按编号顺序读取并堆叠 state_qpos_*.npy 或 action_qpos_*.npy"""
    files = sorted(
        traj_dir.glob(f"{prefix}_*.npy"),
        key=lambda x: int(x.stem.split("_")[-1])  # 取最后一段数字排序
    )
    if not files:
        print(f"❌ 未找到 {prefix}_*.npy 文件")
        return None

    arrays = [np.load(f) for f in files]
    stacked = np.stack(arrays, axis=0)
    print(f"{prefix}: {len(files)} 个文件, 堆叠形状 {stacked.shape}")
    return stacked

if __name__ == "__main__":
    state_qpos = load_qpos_stack(traj_dir, "state_qpos")
    action_qpos = load_qpos_stack(traj_dir, "action_qpos")

    if state_qpos is not None and action_qpos is not None:
        print(f"\n✅ state_qpos shape: {state_qpos.shape}")
        print(f"✅ action_qpos shape: {action_qpos.shape}")
        if state_qpos.shape[1] != action_qpos.shape[1]:
            print("⚠️ 注意：两者列数不同，可能需要对齐或映射。")
