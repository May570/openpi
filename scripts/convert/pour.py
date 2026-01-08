#!/usr/bin/env python3
"""
转换 banana 任务合并成一个LeRobot数据集
"""
import os
os.environ["HF_LEROBOT_HOME"] = "/share/project/wujiling/datasets/"

import sys
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.convert.pour_to_lerobot import port_A2D, DatasetConfig

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="转换 pour 任务数据集（支持 resume / skip）"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="以追加模式继续写入已有数据集，而不是重新创建",
    )
    parser.add_argument(
        "--skip-episode",
        type=int,
        default=0,
        help="从原始数据中跳过前 N 个已经转换过的 episode（全局计数）",
    )
    return parser.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)

    # 配置参数
    base_dir = Path("/share/project/caomingyu/a2d_data")
    repo_id = "tasks_all/pour"
    print(f'HF_LEROBOT_HOME: {os.environ["HF_LEROBOT_HOME"]}')

    # 数据集配置
    dataset_config = DatasetConfig(
        use_videos=False,  # 使用图像模式，更快
        # mode="image",
        image_writer_processes=40,
        image_writer_threads=1,
    )
    
    print(f"开始转换 pour 任务数据集...")
    print(f"基础目录: {base_dir}")
    print(f"数据集ID: {repo_id}")
    print(f"运行模式: {'resume/追加' if args.resume else 'fresh/重建'}")
    print(f"skip_episode（全局跳过）: {args.skip_episode}")
    
    try:
        ds1 = port_A2D(
            raw_dir=base_dir,
            repo_id=repo_id,
            task = "Pick up a cup, place it in the correct position, and then pour the coffee into the cup.",
            skip_episode=args.skip_episode,
            dir_ids = [3924,3926,3928],
            mode="image",
            resume=args.resume,
            dataset_config=dataset_config,
        )

        print(f"\n转换完成！")
        print(f"总episode数量: {len(ds1)}")  
        
    except Exception as e:
        print(f"转换过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 