#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 episodes_stats.jsonl 计算新的 norm_stats.json

用法：
python compute_norm_stats.py \
    --episodes-stats /share/project/wujiling/data/processed/pour_coffee/meta/episodes_stats.jsonl \
    --output /share/project/wujiling/models/base/pi05_base/assets/pour_coffee/norm_stats.json
"""

import argparse
import json
import math
from typing import Dict, Any

import numpy as np


def aggregate_from_episode_stats(episodes_stats_path: str) -> Dict[str, Any]:
    """
    从 episodes_stats.jsonl 聚合得到：
    - state:  来自 "observation.state"
    - actions: 来自 "action"

    返回结构：
    {
        "state": {
            "mean": [...],
            "std": [...],
            "q01": [...],
            "q99": [...],
        },
        "actions": { ... }
    }
    """
    # 映射到输出 key
    fields = {
        "state": "observation.state",
        "actions": "action",
    }

    # 累加器
    sums = {k: None for k in fields}          # Σ μ_e * n_e
    sums_sq = {k: None for k in fields}       # Σ (σ_e^2 + μ_e^2) * n_e
    total_counts = {k: 0 for k in fields}     # Σ n_e
    global_min = {k: None for k in fields}    # 逐维全局最小值
    global_max = {k: None for k in fields}    # 逐维全局最大值

    with open(episodes_stats_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            stats = obj.get("stats", {})

            for out_key, field_name in fields.items():
                if field_name not in stats:
                    continue

                st = stats[field_name]
                mean = np.array(st["mean"], dtype=float)
                std = np.array(st["std"], dtype=float)
                # 你的文件里 count 是一个 length-1 的 list，比如 [1223]
                count_list = st.get("count", [0])
                if not count_list:
                    continue
                n = int(count_list[0])
                if n == 0:
                    continue

                ep_min = np.array(st["min"], dtype=float)
                ep_max = np.array(st["max"], dtype=float)

                # 初始化对应维度
                if sums[out_key] is None:
                    dim = mean.shape[0]
                    sums[out_key] = np.zeros(dim, dtype=float)
                    sums_sq[out_key] = np.zeros(dim, dtype=float)
                    global_min[out_key] = np.full(dim, np.inf, dtype=float)
                    global_max[out_key] = np.full(dim, -np.inf, dtype=float)

                # 累加 Σ μ_e * n_e
                sums[out_key] += mean * n
                # 累加 Σ (σ_e^2 + μ_e^2) * n_e  用于后面算全局二阶矩
                sums_sq[out_key] += (std ** 2 + mean ** 2) * n
                total_counts[out_key] += n

                # 更新全局 min / max
                global_min[out_key] = np.minimum(global_min[out_key], ep_min)
                global_max[out_key] = np.maximum(global_max[out_key], ep_max)

    result = {}
    # z 值：1% 分位的正态分布 z 分数
    z01 = 2.3263478740408408

    for out_key in fields:
        N = total_counts[out_key]
        if N == 0:
            continue

        # 全局均值
        mean = sums[out_key] / N
        # 全局二阶矩
        m2 = sums_sq[out_key] / N
        # 全局方差 & std
        var = m2 - mean ** 2
        var = np.maximum(var, 0.0)  # 数值稳定性
        std = np.sqrt(var)

        # 正态近似计算 q01/q99
        q01 = mean - z01 * std
        q99 = mean + z01 * std

        # 用全局 min/max 裁剪，防止越界
        q01 = np.maximum(q01, global_min[out_key])
        q99 = np.minimum(q99, global_max[out_key])

        result[out_key] = {
            "mean": mean.tolist(),
            "std": std.tolist(),
            "q01": q01.tolist(),
            "q99": q99.tolist(),
        }

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episodes-stats",
        type=str,
        required=True,
        help="episodes_stats.jsonl 的路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出的 norm_stats.json 路径",
    )
    args = parser.parse_args()

    norm_stats = aggregate_from_episode_stats(args.episodes_stats)

    out_obj = {"norm_stats": norm_stats}

    with open(args.output, "w") as f:
        json.dump(out_obj, f, indent=2)

    print(f"[INFO] 写入 norm_stats 到: {args.output}")
    for k, v in norm_stats.items():
        print(f"[INFO] {k}: dim={len(v['mean'])}, total_count=??? (见 episodes_stats)")


if __name__ == "__main__":
    main()
