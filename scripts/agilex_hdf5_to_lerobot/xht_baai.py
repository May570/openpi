#!/usr/bin/env python3
"""
转换Orange任务的脚本
将三个任务文件夹合并成一个LeRobot数据集
"""
import os
os.environ["HF_LEROBOT_HOME"] = "/share/project/wujiling/openpi/datasets/"

import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.agilex_hdf5_to_lerobot.xht_to_lerobot import port_aloha, DatasetConfig

def main():
    # 配置参数
    base_dir = Path("/share/project/wujiling/data")
    repo_id = "tasks/spell_baai"
    print(f'HF_LEROBOT_HOME: {os.environ["HF_LEROBOT_HOME"]}')
    # 数据集配置
    dataset_config = DatasetConfig(
        use_videos=False,  # 使用图像模式，更快
        # mode="image",
        image_writer_processes=5,
        image_writer_threads=3,
    )
    
    print(f"开始转换 spell baai 任务数据集...")
    print(f"基础目录: {base_dir}")
    print(f"数据集ID: {repo_id}")
    
    try:
        task_dir = base_dir / "raw"
        ds1 = port_aloha(
            raw_dir=task_dir,
            repo_id=repo_id,
            task = "Spell BAAI.",
            push_to_hub=False,
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