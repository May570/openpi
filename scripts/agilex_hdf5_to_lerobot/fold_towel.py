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
sys.path.append(str(Path(__file__).parent.parent.parent))  # /share/project/wujiling
# /share/project/wujiling/openpi/scripts/agilex_hdf5_to_lerobot/orange_200_9_8.py

from scripts.agilex_hdf5_to_lerobot.to_lerobot import port_task_with_subtasks, DatasetConfig

def main():
    # 配置参数
    base_dir = Path("/share/project/section/agilex_6_data/agilex_10_data")
    repo_id = "tasks"
    print(f'HF_LEROBOT_HOME: {os.environ["HF_LEROBOT_HOME"]}')
    # 数据集配置
    dataset_config = DatasetConfig(
        use_videos=False,  # 使用图像模式，更快
        # mode="image",
        image_writer_processes=5,
        image_writer_threads=3,
    )
    
    print(f"开始转换fold_towel任务数据集...")
    print(f"基础目录: {base_dir}")
    print(f"数据集ID: {repo_id}")
    
    try:

        task_dir = base_dir / "fold_towel"
        sub_prompt_map = {
            "粉色毛巾": "Fold the pink towel and place it on the tray in the top-left corner.",
            "黄色毛巾": "Fold the yellow towel and place it on the tray in the top-left corner.",
            "灰色毛巾": "Fold the gray towel and place it on the tray in the top-left corner.",
            "蓝白毛巾": "Fold the blue and white towel and place it on the tray in the top-left corner.",
        }
        ds1 = port_task_with_subtasks(
            task_dir=task_dir,
            repo_id=repo_id,
            push_to_hub=False,
            mode="image",
            dataset_config=dataset_config,
            sub_prompt_map=sub_prompt_map,
        )

        print(f"\n转换完成！")
        # AttributeError: 'LeRobotDataset' object has no attribute 'repo_path'
        # print(f"数据集位置: {dataset.repo_path}")
        print(f"总episode数量: {len(ds1)}")  

        # # 批量处理所有任务
        # dataset = port_multiple_tasks(
        #     base_dir=base_dir,
        #     repo_id=repo_id,
        #     push_to_hub=False,  # 先不上传到hub
        #     mode="image",
        #     dataset_config=dataset_config,
        # )
        
        # print(f"\n转换完成！")
        # # AttributeError: 'LeRobotDataset' object has no attribute 'repo_path'
        # # print(f"数据集位置: {dataset.repo_path}")
        # print(f"总episode数量: {len(dataset)}")
        
    except Exception as e:
        print(f"转换过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 