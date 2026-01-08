#!/usr/bin/env python3
"""
统计指定 task_id 范围内的轨迹数量（从标注文件统计）
"""
import os
import json
import argparse

DATA_DIR = "/share/project/caomingyu/a2d_data"

# 要过滤的 task_id 列表（基于 task_id 过滤，而不是 job_id/episode_id）
# 例如: drop_list = {1901, 1902} 会过滤掉 task_id 为 1901 和 1902 的所有轨迹
drop_list = set([1765, 1790, 1830, 1832])


def count_trajectories_in_range(start_id, end_id, data_dir=DATA_DIR):
    """统计指定范围内的有效轨迹数量（从标注文件统计）"""
    print(f"扫描目录: {data_dir}")
    print(f"Job ID 范围: [{start_id}, {end_id}]")
    print("=" * 60)

    valid_count = 0
    job_id_count = {}

    for job_id_str in sorted(os.listdir(data_dir)):
        try:
            job_id_num = int(job_id_str)
            if start_id <= job_id_num <= end_id:
                if job_id_num in drop_list or job_id_str in drop_list:
                    print(f"跳过 Job ID {job_id_str} (在过滤列表中)")
                    continue

                job_dir = os.path.join(data_dir, job_id_str)
                if not os.path.isdir(job_dir):
                    continue

                # 加载标注文件 {job_id}.json
                annotation_file = os.path.join(job_dir, f"{job_id_str}.json")
                if os.path.exists(annotation_file):
                    try:
                        with open(annotation_file, "r", encoding="utf-8") as f:
                            task_list = json.load(f)  # 标注文件是数组格式
                            episode_count = len(task_list)
                            valid_count += episode_count
                            job_id_count[job_id_str] = episode_count
                            if episode_count > 0:
                                print(
                                    f"Job ID {job_id_str}: {episode_count} trajectories"
                                )
                    except Exception as e:
                        print(f"⚠️  加载标注文件失败 {annotation_file}: {e}")
                else:
                    # 标注文件不存在，跳过
                    print(
                        f"⚠️  标注文件不存在: {annotation_file}，跳过 Job ID {job_id_str}"
                    )

        except (ValueError, OSError) as e:
            # 跳过非数字的目录名或其他错误
            continue

    print("=" * 60)
    print(f"有效轨迹数: {valid_count}")
    if drop_list:
        print(f"已过滤的 task_id: {sorted(drop_list)}")
    print("=" * 60)

    return valid_count


def main():
    parser = argparse.ArgumentParser(description="统计指定 job_id 范围内的轨迹数量")
    parser.add_argument("--start", type=int, required=True, help="起始 job_id（包含）")
    parser.add_argument("--end", type=int, required=True, help="结束 job_id（包含）")
    parser.add_argument("--dir", type=str, default=DATA_DIR, help="数据目录路径")

    args = parser.parse_args()

    if args.start > args.end:
        print(f"错误: 起始ID不能大于结束ID")
        return

    count_trajectories_in_range(args.start, args.end, args.dir)


if __name__ == "__main__":
    main()

# 使用示例:
# python3 count_trajectories_range.py --start 2218 --end 2575  # 倒水
# python3 count_trajectories_range.py --start 2846 --end 3112  # 抽屉
# python3 count_trajectories_range.py --start 3149 --end 3153  # 擦黑板
