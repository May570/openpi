#!/usr/bin/env python3
"""
转换 banana 任务合并成一个LeRobot数据集
"""
import os
os.environ["HF_LEROBOT_HOME"] = "/share/project/wujiling/datasets/"

import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.convert.A2D_to_lerobot_4045 import port_A2D, DatasetConfig

def main():
    # 配置参数
    base_dir = Path("/share/project/caomingyu/a2d_data")
    repo_id = "tasks/drawer_4045"
    print(f'HF_LEROBOT_HOME: {os.environ["HF_LEROBOT_HOME"]}')
    # 数据集配置
    dataset_config = DatasetConfig(
        use_videos=False,  # 使用图像模式，更快
        # mode="image",
        image_writer_processes=40,
        image_writer_threads=1,
    )
    
    print(f"开始转换 drawer_4045 任务数据集...")
    print(f"基础目录: {base_dir}")
    print(f"数据集ID: {repo_id}")
    
    try:
        ds1 = port_A2D(
            raw_dir=base_dir,
            repo_id=repo_id,
            task = "Open the drawer, then pick up the item closest to the apple and put it into the drawer, and finally close the drawer.",
            dir_ids = [4045],
            mode="image",
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