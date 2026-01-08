#!/usr/bin/env python3
"""
无限重试版本：
只要 pour.py 没完全执行成功，就会自动 resume + skip-episode 重试。
直到数据集 episode 数不再变化，说明已经全部处理完成。

用法：
    python run_pour_with_resume.py
"""

import os
import sys
import time
import subprocess
from pathlib import Path

os.environ["HF_LEROBOT_HOME"] = "/share/project/wujiling/datasets/"

# 让我们可以正常 import LeRobotDataset
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.convert.pour_to_lerobot import DatasetConfig
from lerobot.common.datasets.lerobot_dataset import (
    HF_LEROBOT_HOME as LEROBOT_HOME,
    LeRobotDataset,
)

REPO_ID = "tasks_open/drawer"

SLEEP_SECONDS = 5       # 每次重试前等待 5 秒
MAX_STAGNANT_RETRY = 3  # 如果连续3次 episode 数不变化 → 认为已完成


def count_existing_episodes():
    """统计当前数据集已写入的 episode 数，若未创建则为 0"""
    output_path = LEROBOT_HOME / REPO_ID
    if not output_path.exists():
        print(f"[resume] 数据集目录不存在，视为 0 条 episode")
        return 0

    cfg = DatasetConfig()
    try:
        dataset = LeRobotDataset(
            repo_id=REPO_ID,
            tolerance_s=cfg.tolerance_s,
            download_videos=False,
        )
    except Exception:
        print("[resume] 打不开数据集，视为 0 条 episode")
        return 0

    try:
        num_eps = dataset.num_episodes
    except AttributeError:
        num_eps = len(dataset)

    print(f"[resume] 当前数据集已有 {num_eps} 条 episode")
    return int(num_eps)


def run_open(resume: bool, skip: int) -> int:
    """执行一次 open.py，返回 returncode"""
    script = Path(__file__).parent / "open.py"
    if not script.exists():
        print(f"找不到 open.py: {script}")
        sys.exit(1)

    cmd = [sys.executable, str(script)]
    if resume:
        cmd.append("--resume")
        cmd.append(f"--skip-episode={skip}")

    print(f"\n====== 执行: {' '.join(cmd)} ======\n")
    p = subprocess.run(cmd)
    return p.returncode


def main():
    stagnant = 0
    last_episode = -1

    print("\n============================")
    print(" 无限重试模式启动")
    print("============================\n")

    while True:
        # 统计目前的 episode 数
        current = count_existing_episodes()

        # 判断是否完全收敛（连续多次不变 → 认为完成）
        if current == last_episode:
            stagnant += 1
            if stagnant >= MAX_STAGNANT_RETRY:
                print("\n🎉 看起来所有 episode 都已成功转换！退出。")
                return
        else:
            stagnant = 0

        last_episode = current

        # 第一次：fresh 模式（如果已经存在就会自动 resume）
        if current == 0:
            print("第一次运行（fresh 模式）")
            code = run_open(resume=False, skip=0)
        else:
            print(f"尝试 resume（skip={current}）")
            code = run_open(resume=True, skip=current)

        # 如果正常完成，看看 episode 是否继续增加
        if code == 0:
            print("本轮正常结束，检查是否还有未转换的 episode...")
        else:
            print("⚠️ 本轮运行异常终止，将自动重试...")

        time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    main()
