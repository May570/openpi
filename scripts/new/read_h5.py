#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用法：
  python inspect_h5.py /path/to/episode_1.hdf5              # 基本信息
  python inspect_h5.py /path/to/episode_1.hdf5 --peek 3     # 每个一维/二维/三维数据集偷看前3条
  python inspect_h5.py /path/to/episode_1.hdf5 --peek 1 --long-names  # 不截断长名字
"""
import argparse, textwrap
import numpy as np
import h5py

def human_shape(shape):
    return "(" + ", ".join(str(s) for s in shape) + ")"

def summarize_array(a, max_items=8):
    flat = a.ravel()
    n = min(len(flat), max_items)
    return np.array2string(flat[:n], threshold=max_items, edgeitems=2)

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("h5", help="episode_x.hdf5 文件路径")
    ap.add_argument("--peek", type=int, default=0, help="每个数据集偷看前N条（对<=3维生效）")
    ap.add_argument("--max", dest="max_print", type=int, default=200, help="最多打印多少个数据集（防爆屏）")
    ap.add_argument("--long-names", action="store_true", help="不截断数据集路径")
    args = ap.parse_args()

    with h5py.File(args.h5, "r") as f:
        # 1) 打印根属性
        print("=== File attributes ===")
        if f.attrs:
            for k, v in f.attrs.items():
                print(f"@{k}: {v}")
        else:
            print("(no file-level attributes)")
        print()

        count = 0
        print("=== Datasets (路径 -> 形状/类型/压缩/块) ===")
        def visit(name, obj):
            nonlocal count
            if isinstance(obj, h5py.Dataset):
                if count >= args.max_print:
                    return
                count += 1
                path = name if args.long_names else (name if len(name) < 120 else name[:117] + "...")
                shape = human_shape(obj.shape)
                dtype = str(obj.dtype)
                compression = obj.compression or "none"
                chunks = obj.chunks
                # 判断“步长维度”可能是哪一维（通常 T 在第0维）
                time_dim_hint = "T=?"
                if len(obj.shape) >= 1:
                    time_dim_hint = f"T≈{obj.shape[0]}"
                print(f"{path}: shape={shape}, dtype={dtype}, compression={compression}, chunks={chunks}, {time_dim_hint}")

                # 2) 数据集级别属性（如果有）
                if obj.attrs:
                    for ak, av in obj.attrs.items():
                        print(f"  - attr @{ak}: {av}")

                # 3) 可选偷看前 N 条（限维度<=3，防止巨量输出）
                if args.peek > 0 and obj.ndim <= 3 and obj.size > 0:
                    try:
                        n = min(args.peek, obj.shape[0] if obj.ndim >= 1 else 1)
                        sample = obj[0:n]
                        # 如果是图像类数据，尽量只打印尺寸与极值
                        if sample.dtype == np.uint8 and sample.ndim in (3,4):
                            s = sample.shape
                            print(f"📊  -> peek[{n}] uint8, sample.shape={s}, min={sample.min()}, max={sample.max()}")
                        else:
                            print(f"📊  -> peek[{n}]: {summarize_array(sample)}")
                    except Exception as e:
                        print(f"  -> peek error: {e}")
            # 如果是 group，打印一下直接子项个数，方便定位结构
            elif isinstance(obj, h5py.Group):
                # 只对顶层和二级分组做一点提示
                depth = name.count("/")
                if depth <= 2:
                    try:
                        keys = list(obj.keys())
                        print(f"[📁 GROUP] {name or '/'} (children={len(keys)}): " +
                              (", ".join(keys[:8]) + ("..." if len(keys) > 8 else "")))
                    except Exception:
                        pass

        f.visititems(visit)

        print("\n=== Summary ===")
        print(f"Total datasets printed: {min(count, args.max_print)}"
              + (f" (truncated at --max={args.max_print})" if count >= args.max_print else ""))

if __name__ == "__main__":
    main()
