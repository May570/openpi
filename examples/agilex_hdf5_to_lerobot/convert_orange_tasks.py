#!/usr/bin/env python3
"""
转换Orange任务的脚本
将三个任务文件夹合并成一个LeRobot数据集
"""
import os
os.environ["HF_LEROBOT_HOME"] = "/share/project/section/"

import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent.parent))  # /share/project/lvhuaihai
# /share/project/lvhuaihai/openpi/examples/agilex_hdf5_to_lerobot/convert_orange_tasks.py

from convert_agilex_to_lerobot import port_multiple_tasks, DatasetConfig

def main():
    # 配置参数
    base_dir = Path("/share/project/section/task/task_orange_200_8.11")
    # task_orange_110_8.11
    # task_orange_30_8.11
    # task_orange_60_8.11
    repo_id = "task/orange_tasks"
    print(f'HF_LEROBOT_HOME: {os.environ["HF_LEROBOT_HOME"]}')
    # 数据集配置
    dataset_config = DatasetConfig(
        use_videos=False,  # 使用图像模式，更快
        # mode="image",
        image_writer_processes=5,
        image_writer_threads=3,
    )
    
    print(f"开始转换Orange任务数据集...")
    print(f"基础目录: {base_dir}")
    print(f"数据集ID: {repo_id}")
    
    try:
        # 批量处理所有任务
        dataset = port_multiple_tasks(
            base_dir=base_dir,
            repo_id=repo_id,
            push_to_hub=False,  # 先不上传到hub
            mode="image",
            dataset_config=dataset_config,
        )
        
        print(f"\n转换完成！")
        # AttributeError: 'LeRobotDataset' object has no attribute 'repo_path'
        # print(f"数据集位置: {dataset.repo_path}")
        print(f"总episode数量: {len(dataset)}")
        
    except Exception as e:
        print(f"转换过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 