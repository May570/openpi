#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Modified version: only keep BAAI part of each raw H5,
output standalone episode_xxx_baai.h5 files,
no JSON needed.
"""

import os
import json
import argparse
import h5py
import numpy as np
from tqdm import tqdm

# ------------------------------------------------------
# Convert RAW.h5 → baai_only episode.h5
# ------------------------------------------------------
def convert_baai_episode(in_path, out_path, end_frame):
    """
    只保留 BAAI 段，并且在该段内：
    - 从 start_idx=1 开始（跳过 index 0）
    - 扫描 joint/gripper/pose 多个通道
    - 遇到 shape 异常的帧就跳过，只保留所有通道 shape 都正常的帧
    """
    print(f"[INFO] Converting BAAI portion of: {in_path}")
    print(f"[INFO] BAAI end frame (from anno) = {end_frame}")

    with h5py.File(in_path, "r") as f_in, h5py.File(out_path, "w") as f_out:

        start_idx = 1
        raw_len = len(f_in["/joint_left"])
        end_idx = min(int(end_frame), raw_len)

        # -------- 1. 找一个“参考帧”，用来确定各通道的正常 shape --------
        #   参考通道：joint_left/right, gripper_left/right, pose_left/right
        ds_names = [
            ("/joint_left",   "joint_left"),
            ("/joint_right",  "joint_right"),
            ("/gripper_left", "gripper_left"),
            ("/gripper_right","gripper_right"),
        ]
        dsets = {}
        for path, name in ds_names:
            if path not in f_in:
                print(f"[WARN] {path} not found in {in_path}, skip this file.")
                return
            dsets[name] = f_in[path]

        ref_idx = None
        expected_shapes = {}
        for i in range(start_idx, end_idx):
            ok = True
            for name, dset in dsets.items():
                shp = dset[i].shape
                if len(shp) == 0:   # 空 array 视为坏帧
                    ok = False
                    break
            if ok:
                ref_idx = i
                for name, dset in dsets.items():
                    expected_shapes[name] = dset[ref_idx].shape
                break

        if ref_idx is None:
            print(f"[WARN] No valid reference frame in {in_path}, skip this file.")
            return

        print(f"[INFO] Use frame {ref_idx} as reference, expected shapes:")
        for name, shp in expected_shapes.items():
            print(f"       {name}: {shp}")

        # -------- 2. 在 [start_idx, end_idx) 之间收集所有“好帧”索引 --------
        valid_idxs = []
        bad_count = 0

        for i in range(start_idx, end_idx):
            bad = False
            for name, dset in dsets.items():
                if dset[i].shape != expected_shapes[name]:
                    bad = True
                    bad_count += 1
                    # 可选打印详细信息：
                    # print(f"[WARN] {name}[{i}] shape {dset[i].shape} != {expected_shapes[name]}")
                    break
            if not bad:
                valid_idxs.append(i)

        if len(valid_idxs) == 0:
            print(f"[WARN] No valid frames after filtering in {in_path}, skip.")
            return

        print(f"[INFO] Collected {len(valid_idxs)} valid frames, skipped {bad_count} bad frames.")

        T = len(valid_idxs)

        # -------- 3. 复制图像：用同一个 valid_idxs 保证对齐 --------
        cam_map = {
            "image_right": "cam_high",
            "image_left_wrist": "cam_left_wrist",
            "image_right_wrist": "cam_right_wrist",
        }

        for src_name, dst_name in cam_map.items():
            src_path = f"/observations/images/{src_name}"
            if src_path not in f_in:
                print(f"[WARN] {src_path} not in file, skip.")
                continue
            dset = f_in[src_path]
            data = [dset[i] for i in valid_idxs]
            f_out.create_dataset(
                f"/observations/images/{dst_name}",
                data=data,
                dtype=dset.dtype,
            )

        # -------- 4. ACTION & STATE：统一用 valid_idxs 索引 --------
        action = f_out.create_group("action")
        state = f_out.create_group("state")

        # helper：从 dset 里按 valid_idxs 收集 stack
        def stack_by_indices(dset, idxs):
            return np.stack([dset[i] for i in idxs])

        # qpos
        qpos_l = stack_by_indices(dsets["joint_left"],  valid_idxs)[:, :6]
        qpos_r = stack_by_indices(dsets["joint_right"], valid_idxs)[:, :6]

        # gripper
        gr_l = stack_by_indices(dsets["gripper_left"],  valid_idxs)
        gr_r = stack_by_indices(dsets["gripper_right"], valid_idxs)

        for root in (action, state):
            g_qpos = root.create_group("qpos")
            g_qpos.create_dataset("left",  data=qpos_l)
            g_qpos.create_dataset("right", data=qpos_r)

            g_gr = root.create_group("gripper")
            g_gr.create_dataset("left",  data=gr_l)
            g_gr.create_dataset("right", data=gr_r)

        # -------- 5. 元数据 --------
        meta = f_out.create_group("episode_metadata")
        meta.create_dataset("is_first", data=np.array([True] + [False] * (T - 1)))
        meta.create_dataset("is_last",  data=np.array([False] * (T - 1) + [True]))

        f_out.attrs["episode_length"] = T
        f_out.attrs["data_format"] = "baai_only_converted"

    print(f"[OK] Saved BAAI-only episode → {out_path}")


# ------------------------------------------------------
# Main pipeline
# ------------------------------------------------------
def main(args):
    # Load annotation file
    with open(args.anno_file, "r") as f:
        anns = json.load(f)

    # Build filename → annotation index
    ann_map = {
        os.path.basename(a["path"]): a
        for a in anns
    }

    # Collect all raw h5 in directory list
    all_h5 = []
    for d in args.file_dir_list:
        for root, _, files in os.walk(d):
            for f in files:
                if f.endswith(".h5") or f.endswith(".hdf5"):
                    all_h5.append(os.path.join(root, f))

    os.makedirs(args.base_path, exist_ok=True)

    counter = 0

    for raw_path in tqdm(all_h5):

        fname = os.path.basename(raw_path)
        if fname not in ann_map:
            print(f"[SKIP] No annotation for: {fname}")
            continue

        ann = ann_map[fname]
        videoLabels = ann["videoLabels"]

        # ------------------------------------
        # Find *first* "End" → BAAI end frame
        # ------------------------------------
        baai_end = None
        for item in videoLabels:
            if item["timelinelabels"][0] == "End":
                baai_end = item["ranges"][0]["end"]
                break

        if baai_end is None:
            print(f"[WARN] No 'End' label for {fname}, skip.")
            continue

        out_name = f"episode_{counter:06d}_baai.h5"
        out_path = os.path.join(args.base_path, out_name)

        convert_baai_episode(
            in_path=raw_path,
            out_path=out_path,
            end_frame=baai_end  # already 1-based → direct slice works (0~end-1)
        )

        counter += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_dir_list", nargs="+", required=True,
                        help="Dirs containing raw H5 files")
    parser.add_argument("--anno_file", type=str, required=True,
                        help="Annotation JSON file")
    parser.add_argument("--base_path", type=str, required=True,
                        help="Output directory for baai-only episodes")
    args = parser.parse_args()

    # args = argparse.Namespace(
    #     file_dir_list="/share/project/guowei/data/demo/XHT/XHT1/task_pin_baai_zai_pin_brain_XHT1",
    #     anno_file="/share/project/guowei/data/demo/XHT/XHT1/task_pin_baai_zai_pin_brain_XHT1/task_pin_baai_zai_pin_brain_XHT1_yz.json",
    #     base_path="/share/project/wujiling/data/raw/"
    # )

    main(args)
