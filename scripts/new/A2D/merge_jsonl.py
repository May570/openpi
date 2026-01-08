#!/usr/bin/env python3
"""
合并6D版本的两套JSONL文件（320x240和640x480）
处理完成后删除原分片文件
"""
import os
import glob
import json
from pathlib import Path
from tqdm import tqdm


def merge_jsonl_by_resolution(data_dir, task_name, resolution):
    """
    合并指定分辨率的所有JSONL分片

    Args:
        data_dir: JSONL文件所在目录
        task_name: 任务名称（倒水/抽屉/擦黑板）
        resolution: 分辨率标识，如 "320x240" 或 "640x480"
    """
    data_dir = Path(data_dir)
    
    # 查找任务目录下的分片JSONL文件
    task_dir = data_dir / task_name
    pattern = str(task_dir / f"a2d_train_{task_name}_{resolution}_*_*.jsonl")
    jsonl_files = sorted(glob.glob(pattern))

    if not jsonl_files:
        print(f"❌ 未找到任何 {resolution} 的JSONL文件: {pattern}")
        return False

    print(f"\n{'='*60}")
    print(f"📊 处理 {resolution} 版本")
    print(f"{'='*60}")
    print(f"找到 {len(jsonl_files)} 个分片文件")

    # 统计原始文件信息
    total_lines = 0
    file_info = []

    for file_path in jsonl_files:
        with open(file_path, "r", encoding="utf-8") as f:
            line_count = sum(1 for _ in f)
            total_lines += line_count
            file_info.append((Path(file_path).name, line_count))

    print(f"\n📝 原始文件统计:")
    for name, count in file_info[:3]:
        print(f"   {name}: {count} 行")
    if len(file_info) > 6:
        print(f"   ... (共 {len(file_info)} 个文件)")
        for name, count in file_info[-3:]:
            print(f"   {name}: {count} 行")
    else:
        for name, count in file_info[3:]:
            print(f"   {name}: {count} 行")
    print(f"\n✅ 总行数: {total_lines}")

    # 合并文件
    output_filename = f"a2d_train_{task_name}_{resolution}_merged.jsonl"
    output_path = task_dir / output_filename
    temp_output_path = task_dir / f"{output_filename}.tmp"

    print(f"\n🔄 开始合并文件...")
    print(f"   输出文件: {output_path}")

    merged_count = 0

    try:
        with open(temp_output_path, "w", encoding="utf-8") as out_f:
            for file_path in tqdm(jsonl_files, desc="合并进度"):
                with open(file_path, "r", encoding="utf-8") as in_f:
                    for line in in_f:
                        line = line.strip()
                        if line:
                            try:
                                json.loads(line)
                                out_f.write(line + "\n")
                                merged_count += 1
                            except json.JSONDecodeError as e:
                                print(
                                    f"\n⚠️  跳过无效JSON行 in {Path(file_path).name}: {e}"
                                )

        # 重命名临时文件为最终文件
        temp_output_path.rename(output_path)

        print(f"\n✅ 合并完成!")
        print(f"   合并行数: {merged_count}")
        print(f"   预期行数: {total_lines}")

        if merged_count != total_lines:
            print(f"⚠️  行数不匹配！差异: {total_lines - merged_count}")

        # 验证合并后的文件
        print(f"\n🔍 验证合并文件...")
        with open(output_path, "r", encoding="utf-8") as f:
            verify_count = sum(1 for _ in f)

        if verify_count == merged_count:
            print(f"✅ 验证通过: {verify_count} 行")
        else:
            print(f"❌ 验证失败: 写入{merged_count}行, 实际{verify_count}行")
            return False

        # 删除原始分片文件
        print(f"\n🗑️  删除原始分片文件...")
        deleted_count = 0
        for file_path in tqdm(jsonl_files, desc="删除进度"):
            try:
                os.remove(file_path)
                deleted_count += 1
            except Exception as e:
                print(f"\n⚠️  删除失败 {Path(file_path).name}: {e}")

        print(f"✅ 已删除 {deleted_count}/{len(jsonl_files)} 个原始文件")
        print(f"\n🎉 {resolution} 版本处理完成!")
        print(f"   最终文件: {output_path}")
        print(f"   总行数: {verify_count}")
        print(f"   文件大小: {output_path.stat().st_size / (1024**2):.2f} MB")

        return True

    except Exception as e:
        print(f"\n❌ 合并失败: {e}")
        if temp_output_path.exists():
            temp_output_path.unlink()
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description="合并A2D任务的分片JSONL文件")
    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        choices=["pour_coffee_105_", "open_close", "erase", "pnp"],
        help="任务类型: pour (倒水), open_close (开关抽屉), erase (擦黑板), pnp(抓取放置)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/share/project/fengli/data",
        help="JSONL文件所在目录",
    )
    parser.add_argument("--dry-run", action="store_true", help="试运行，只统计不合并")

    args = parser.parse_args()

    if args.dry_run:
        print(f"🔍 试运行模式（不会修改文件）")
        print(f"任务类型: {args.task_name}")
        data_dir = Path(args.data_dir)

        for resolution in ["320x240", "640x480"]:
            pattern = str(
                data_dir / f"a2d_train_{args.task_name}_{resolution}_*_*.jsonl"
            )
            jsonl_files = sorted(glob.glob(pattern))

            print(f"\n{resolution} 找到 {len(jsonl_files)} 个文件:")
            total = 0
            for f in jsonl_files[:5]:
                with open(f, "r") as file:
                    count = sum(1 for _ in file)
                    total += count
                    print(f"  {Path(f).name}: {count} 行")
            if len(jsonl_files) > 5:
                print(f"  ... (共 {len(jsonl_files)} 个文件)")
            print(f"总计: {total} 行 (已统计前5个)")
    else:
        print("=" * 60)
        print(f"🚀 开始合并【{args.task_name}】任务的JSONL文件")
        print("=" * 60)

        # 合并320x240版本
        success_320 = merge_jsonl_by_resolution(
            args.data_dir, args.task_name, "320x240"
        )

        # 合并640x480版本
        success_640 = merge_jsonl_by_resolution(
            args.data_dir, args.task_name, "640x480"
        )

        print(f"\n{'='*60}")
        print("📊 最终统计")
        print(f"{'='*60}")
        print(f"320x240 版本: {'✅ 成功' if success_320 else '❌ 失败'}")
        print(f"640x480 版本: {'✅ 成功' if success_640 else '❌ 失败'}")

        if success_320 and success_640:
            print(f"\n🎉 全部完成！")
            print(f"\n最终生成文件:")
            print(
                f"  - {args.data_dir}/a2d_train_{args.task_name}_320x240_merged.jsonl"
            )
            print(
                f"  - {args.data_dir}/a2d_train_{args.task_name}_640x480_merged.jsonl"
            )
        else:
            print(f"\n⚠️  部分任务失败，请检查日志")


if __name__ == "__main__":
    main()
