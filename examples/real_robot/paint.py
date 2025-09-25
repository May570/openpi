"""分析并可视化机器人动作数据的脚本"""
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

data_dir = "/home/admin123/桌面/wjl/openpi/examples/real_robot/infer_actions1"
files = sorted(glob.glob(os.path.join(data_dir, "chunk_*.npy")))

if not files:
    print("⚠️ 没有找到任何 chunk_*.npy 文件，请检查路径:", data_dir)
else:
    all_overall = []   # 每个 chunk 的整体曲线
    all_dims = []      # list of (chunk, (T-1, D))

    for f in files:
        actions = np.load(f)
        if actions.ndim != 2:
            print(f"⚠️ {f} 不是二维数据，跳过")
            continue

        T, D = actions.shape
        delta = actions[1:] - actions[:-1]  # (T-1, D)

        # 整体 L2 曲线（>=0）
        v_all = np.linalg.norm(delta, axis=1)
        v_all_smooth = savgol_filter(v_all, 7, 2) if len(v_all) >= 7 else v_all
        all_overall.append(v_all_smooth)

        # 每个维度改成绝对值
        delta_abs = np.abs(delta)
        all_dims.append(delta_abs)

    num_chunks = len(all_overall)
    steps = np.arange(1, T)  # 横坐标 (1 ~ T-1)

    # ---------- 图1: 所有 chunk 的整体曲线 ----------
    cols = 4
    rows = (num_chunks + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows), sharex=True, sharey=True)
    axes = axes.flatten()

    # 统一纵坐标范围
    overall_max = max([np.max(v) for v in all_overall])
    ylim = (0, overall_max * 1.2)

    for i, v in enumerate(all_overall):
        axes[i].plot(steps, v, color="blue")
        axes[i].set_title(f"Chunk {i}")
        axes[i].set_ylim(ylim)
        axes[i].set_xticks(np.arange(0, T, 10))  # 每隔10打刻度
        axes[i].grid(True)

    for j in range(num_chunks, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Overall Speed (L2) per Chunk", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_file = os.path.join(data_dir, "all_chunks_overall.png")
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"✅ 已保存 {out_file}")

    # ---------- 图2-15: 每个维度 ----------
    for d in range(D):
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows), sharex=True, sharey=True)
        axes = axes.flatten()

        # 统一纵坐标范围（当前维度）
        dim_max = max([np.max(chunk[:, d]) for chunk in all_dims])
        ylim = (0, dim_max * 1.2)

        for i, chunk in enumerate(all_dims):
            v_dim = chunk[:, d]
            v_dim_smooth = savgol_filter(v_dim, 7, 2) if len(v_dim) >= 7 else v_dim
            axes[i].plot(steps, v_dim_smooth, color="orange")
            axes[i].set_title(f"Chunk {i}")
            axes[i].set_ylim(ylim)
            axes[i].set_xticks(np.arange(0, T, 10))
            axes[i].grid(True)

        for j in range(num_chunks, len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle(f"Dimension {d} |Δa| per Chunk", fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out_file = os.path.join(data_dir, f"all_chunks_dim{d}.png")
        plt.savefig(out_file, dpi=150)
        plt.close()
        print(f"✅ 已保存 {out_file}")
