#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 改这里两三行
H5_PATH = "/share/project/wujiling/data/raw/task_pin_baai_zai_pin_brain_XHT1_plus2_h5s_32/episode_000009_baai.h5"  # ← 改成你的文件
OUT_PATH = "/share/project/wujiling/openpi/scripts/new/cam_high.mp4"     # 设为 None 就自动生成同名 .cam_high.mp4；也可填路径字符串
FPS = 50            # 输出视频帧率
RESIZE = None       # 例: (640, 480)；不缩放就设为 None
LIMIT = None        # 只导出前 N 帧（预览用），不限制就设为 None

# 固定相机键：只导出 cam_high
CAMERA_KEY = "observations/images/cam_high"

# ==== 下面无需改动 ====

import io
from pathlib import Path
import numpy as np
import h5py
from PIL import Image
import cv2

def decode_object_frame(obj_item):
    """将 dtype=object 的单帧图像字节流解码为 RGB ndarray。"""
    if isinstance(obj_item, (bytes, bytearray)):
        buf = obj_item
    elif isinstance(obj_item, np.void):
        buf = obj_item.tobytes()
    else:
        buf = memoryview(obj_item).tobytes()
    # 先用 PIL（兼容 PNG/JPEG）
    try:
        img = Image.open(io.BytesIO(buf))
        img = img.convert("RGB")
        return np.asarray(img)
    except Exception:
        # 退回 OpenCV
        arr = np.frombuffer(buf, dtype=np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("该帧无法解码（可能不是有效PNG/JPEG）。")
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def load_first_frame(ds):
    """读取第一帧以推断分辨率/存储格式。"""
    if ds.dtype == np.object_ or ds.dtype.kind in ("O", "V"):
        return decode_object_frame(ds[0]), "object"
    else:
        arr = ds[0]
        if arr.ndim == 3 and arr.shape[-1] == 3:     # (H,W,3)
            return arr, "uint8_hw3"
        elif arr.ndim == 3 and arr.shape[0] == 3:    # (3,H,W)
            return np.moveaxis(arr, 0, -1), "uint8_chw"
        else:
            raise ValueError(f"不支持的相机帧形状: {arr.shape}")

def frames_count(ds):
    return len(ds) if (ds.dtype == np.object_ or ds.dtype.kind in ("O","V")) else ds.shape[0]

def open_video_writer(out_path: Path, width: int, height: int, fps: int):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 通用 H.264 不可用时可改 "MJPG"
    vw = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not vw.isOpened():
        raise RuntimeError("VideoWriter 打开失败：请检查输出路径或系统编解码器。")
    return vw

def main():
    h5_path = Path(H5_PATH)
    if OUT_PATH is None:
        out_path = h5_path.with_suffix(".cam_high.mp4")
    else:
        out_path = Path(OUT_PATH)

    with h5py.File(h5_path, "r") as f:
        if CAMERA_KEY not in f:
            # 允许简写（只写 cam_high）
            short = CAMERA_KEY.split("/")[-1]
            alt = f"observations/images/{short}"
            if alt in f:
                cam_key = alt
            else:
                # 列出可选键帮助排查
                if "observations" in f and "images" in f["observations"]:
                    print("可用相机键：")
                    for k in f["observations/images"].keys():
                        print(" - observations/images/" + k)
                raise KeyError(f"未找到相机数据集: {CAMERA_KEY}")
        else:
            cam_key = CAMERA_KEY

        ds = f[cam_key]
        first_frame, storage = load_first_frame(ds)
        H, W = first_frame.shape[:2]

        out_W, out_H = (RESIZE if RESIZE else (W, H))
        vw = open_video_writer(out_path, out_W, out_H, FPS)

        total = frames_count(ds)
        if LIMIT is not None:
            total = min(total, LIMIT)

        print(f"[INFO] 导出 {cam_key} => {out_path}")
        print(f"[INFO] 存储: {storage} | 原分辨率: {W}x{H} | 输出: {out_W}x{out_H} | 帧数: {total} | FPS: {FPS}")

        for i in range(total):
            if storage == "object":
                rgb = decode_object_frame(ds[i])
            elif storage == "uint8_hw3":
                rgb = ds[i]
            else:  # "uint8_chw"
                rgb = np.moveaxis(ds[i], 0, -1)

            if (rgb.shape[1], rgb.shape[0]) != (out_W, out_H):
                rgb = cv2.resize(rgb, (out_W, out_H), interpolation=cv2.INTER_AREA)

            vw.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        vw.release()
        print("[OK] 完成。")

if __name__ == "__main__":
    main()
