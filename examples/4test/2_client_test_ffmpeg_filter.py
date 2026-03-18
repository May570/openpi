# episode_framewise_test.py
# 功能：
# 1) 对单个 episode parquet 的每一帧调用推理服务 /infer
# 2) 将每一帧预测到的未来动作序列与 truth 对齐（不足50步则有多少用多少）
# 3) 得到每一帧误差（MSE / RMSE），绘制误差曲线并保存
# 4) 生成 cam_high 视频，并在左上角叠加帧号
#
# 依赖：
# pip install pandas numpy requests pillow matplotlib imageio opencv-python pyarrow
#
# 用法示例：
# python episode_framewise_test.py \
#   --url http://127.0.0.1:5001 \
#   --data-file /path/to/episode_000001.parquet \
#   --out-dir ./out_eval \
#   --action-horizon 50 \
#   --pred-field qpos \
#   --fps 20
# python examples/4test/2_client_test_ffmpeg.py \ 
#   --data-file /share/project/wujiling/datasets/clean/adjust_bottle_demo/data/chunk-000/episode_000001.parquet \
#


import os
import json
import time
import base64
import argparse
import shutil
import subprocess
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import requests

from PIL import Image, ImageDraw, ImageFont

# matplotlib 可选：没有就不画图
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

# video：优先用 imageio 写 mp4（更省心）；没有就尝试用 opencv
try:
    import imageio.v3 as iio
    HAS_IMAGEIO = True
except Exception:
    HAS_IMAGEIO = False

try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False



def _ffmpeg_available() -> bool:
    """Return True if ffmpeg is available in PATH."""
    return shutil.which("ffmpeg") is not None


class FFMPEGVideoWriter:
    """Stream raw BGR frames to ffmpeg to produce a VSCode-friendly MP4 (H.264 yuv420p +faststart)."""

    def __init__(self, out_path: str, fps: int, width: int, height: int):
        self.out_path = out_path
        self.fps = int(fps)
        self.width = int(width)
        self.height = int(height)
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}",
            "-r", str(self.fps),
            "-i", "pipe:0",
            "-an",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            out_path,
        ]
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write(self, frame_bgr: np.ndarray) -> None:
        if self.proc is None or self.proc.stdin is None:
            return
        if frame_bgr is None:
            return
        if frame_bgr.dtype != np.uint8:
            frame_bgr = frame_bgr.astype(np.uint8, copy=False)
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError(f"Expected frame shape (H,W,3), got {frame_bgr.shape}")
        if frame_bgr.shape[1] != self.width or frame_bgr.shape[0] != self.height:
            raise ValueError(
                f"Frame size mismatch. Expected {(self.height, self.width)}, got {frame_bgr.shape[:2]}"
            )
        # Ensure contiguous
        frame_bgr = np.ascontiguousarray(frame_bgr)
        try:
            self.proc.stdin.write(frame_bgr.tobytes())
        except BrokenPipeError:
            err = self.proc.stderr.read().decode("utf-8", errors="replace") if self.proc.stderr else ""
            raise RuntimeError(f"ffmpeg pipe broken. ffmpeg stderr:\n{err}")

    def close(self) -> None:
        if self.proc is None:
            return
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
        except Exception:
            pass
        stderr = ""
        try:
            if self.proc.stderr:
                stderr = self.proc.stderr.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        ret = self.proc.wait()
        if ret != 0:
            # best-effort cleanup
            try:
                if os.path.exists(self.out_path):
                    os.remove(self.out_path)
            except Exception:
                pass
            raise RuntimeError(f"ffmpeg failed with code {ret}. stderr:\n{stderr}")


# -----------------------------
# 工具函数
# -----------------------------
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def safe_to_list(x: Any) -> List[float]:
    """把 pandas/np/pyarrow 里可能出现的 array-like 转成 python list"""
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, (np.ndarray,)):
        return x.tolist()
    # pandas 可能给你个 dtype=object 的东西
    try:
        return list(x)
    except Exception:
        return [x]


def extract_image_bytes(cell: Any) -> Optional[bytes]:
    """
    parquet 中 observation.images.cam_high 这类字段，通常是 dict：
      {"bytes": b"..."}  或 {"bytes": "<already string>"} 或 {"bytes": <pyarrow binary>}
    """
    if cell is None:
        return None
    if isinstance(cell, dict):
        b = cell.get("bytes", None)
    else:
        # 有些数据直接存 bytes
        b = cell

    if b is None:
        return None

    if isinstance(b, bytes):
        return b

    # 可能是 pyarrow Scalar / memoryview / numpy scalar
    if isinstance(b, memoryview):
        return b.tobytes()

    # 如果已经是 base64 字符串（或普通字符串），先尝试 base64 解码；失败就返回 None
    if isinstance(b, str):
        try:
            return base64.b64decode(b)
        except Exception:
            return None

    # 兜底：尝试转 bytes
    try:
        return bytes(b)
    except Exception:
        return None


def bytes_to_pil_image(img_bytes: bytes) -> Optional[Image.Image]:
    if not img_bytes:
        return None
    try:
        from io import BytesIO
        return Image.open(BytesIO(img_bytes)).convert("RGB")
    except Exception:
        return None


def pil_to_base64_str(img: Image.Image, fmt: str = "JPEG") -> str:
    from io import BytesIO
    buf = BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def overlay_frame_idx(img: Image.Image, frame_idx: int) -> Image.Image:
    """左上角叠加帧号"""
    out = img.copy()
    draw = ImageDraw.Draw(out)

    # 尽量用默认字体；环境里没有字体文件也能跑
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 24)
    except Exception:
        font = ImageFont.load_default()

    text = f"Frame: {frame_idx}"
    # 画一个半透明背景块（PIL 没有原生半透明 draw，简单用实色）
    pad = 6
    tw, th = draw.textbbox((0, 0), text, font=font)[2:]
    draw.rectangle([0, 0, tw + 2 * pad, th + 2 * pad], fill=(0, 0, 0))
    draw.text((pad, pad), text, fill=(255, 255, 255), font=font)
    return out


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


# -----------------------------
# 主类：逐帧评估器
# -----------------------------
class EpisodeFramewiseTester:
    def __init__(
        self,
        url: str,
        data_file: str,
        out_dir: str,
        action_horizon: int = 50,
        pred_field: str = "qpos",
        prompt: str = " ",
        timeout_s: int = 100,
        sleep_s: float = 0.0,
        fps: int = 20,
        video_limit: Optional[int] = None,
        use_images: Tuple[str, ...] = ("cam_high", "cam_left_wrist", "cam_right_wrist", "image", "wrist_image"),
    ):
        self.url = url.rstrip("/")
        self.data_file = data_file
        self.out_dir = out_dir
        self.action_horizon = int(action_horizon)
        self.pred_field = pred_field
        self.prompt = prompt
        self.timeout_s = timeout_s
        self.sleep_s = float(sleep_s)
        self.fps = int(fps)
        self.video_limit = video_limit
        self.use_images = use_images

        ensure_dir(self.out_dir)
        ensure_dir(os.path.join(self.out_dir, "per_frame_results"))
        ensure_dir(os.path.join(self.out_dir, "figs"))

        self.df = pd.read_parquet(self.data_file)
        self.T = len(self.df)

        # 基本列名（和你原脚本一致的约定）
        self.col_state = "observation.state"
        self.col_action = "action"
        # self.col_state = "state"
        # self.col_action = "actions"
        self.col_img_prefix = "observation.images."

        # 结果容器
        self.frame_errors_mse: List[float] = []
        self.frame_errors_rmse: List[float] = []
        self.frame_valid_steps: List[int] = []
        self.http_ok: List[bool] = []
        self.latencies: List[Optional[float]] = []


        # -----------------------------
        # 静止帧跳过推理（从第2帧开始）
        # 判断依据：cam_high 画面变化 + observation.state(手臂)变化
        # 阈值经验设定：你可根据数据噪声再微调
        # -----------------------------
        self.static_img_mad_threshold = 2.0   # 图像平均绝对差 (0~255) 小于该值视为“几乎不变”
        self.static_state_mad_threshold = 1e-3  # state 平均绝对差 小于该值视为“几乎不变”
        self.static_state_mad_threshold_loose = 6e-3
        self.static_img_downsample = 64  # 计算图像差时先缩放到 64x64 以加速


    def test_service_info(self) -> bool:
        try:
            r = requests.get(f"{self.url}/info", timeout=10)
            if r.status_code != 200:
                print(f"[ERROR] /info status={r.status_code} body={r.text[:200]}")
                return False
            print("[OK] /info:", r.json())
            return True
        except Exception as e:
            print(f"[ERROR] cannot reach /info: {e}")
            return False

    def build_payload_for_frame(self, i: int) -> Dict[str, Any]:
        row = self.df.iloc[i]
        # print(row)

        # state
        state = safe_to_list(row.get(self.col_state, []))
        payload: Dict[str, Any] = {
            "state": [state],  # 保持与原脚本一致：二维
            "eef_pose": [[0, 0, 0, 0, 0, 0, 0]],  # 占位（按你服务端 schema 必填与否自行改）
            "images": {},
            "prompt": self.prompt,
        }

        # images: base64
        for cam in self.use_images:
            key = f"{self.col_img_prefix}{cam}"
            # key = f"{cam}"
            cell = row.get(key, None)
            img_bytes = extract_image_bytes(cell)
            if img_bytes is None:
                continue
            payload["images"][cam] = base64.b64encode(img_bytes).decode("utf-8")

        return payload

    def get_truth_action_seq(self, i: int) -> np.ndarray:
        """
        truth 未来动作：从 i 到 i+H-1，有多少取多少
        shape: [L, D]
        """
        end = min(i + self.action_horizon, self.T)
        seq = []
        for k in range(i, end):
            a = safe_to_list(self.df.iloc[k].get(self.col_action, []))
            seq.append(a)
        return np.asarray(seq, dtype=np.float32)

    def infer(self, payload: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], Optional[float]]:
        t0 = time.time()
        try:
            r = requests.post(f"{self.url}/infer", json=payload, timeout=self.timeout_s)
            latency = time.time() - t0
            if r.status_code != 200:
                return False, {"error": f"http {r.status_code}", "body": r.text}, latency
            data = r.json()
            if not bool(data.get("success", False)):
                return False, data, latency
            return True, data, latency
        except Exception as e:
            latency = time.time() - t0
            return False, {"error": str(e)}, latency

    def extract_pred_seq(self, infer_json: Dict[str, Any]) -> np.ndarray:
        """
        从服务端返回取预测动作序列。
        默认字段 pred_field=qpos（与你原脚本一致）。
        期望 shape: [L, D]
        """
        seq = infer_json.get(self.pred_field, [])
        if seq is None:
            seq = []
        return np.asarray(seq, dtype=np.float32)


    def _get_state_vec(self, i: int) -> np.ndarray:
        row = self.df.iloc[i]
        state = safe_to_list(row.get(self.col_state, []))
        return np.asarray(state, dtype=np.float32).reshape(-1)

    def _get_cam_high_bytes(self, i: int) -> Optional[bytes]:
        row = self.df.iloc[i]
        key = f"{self.col_img_prefix}cam_high"
        cell = row.get(key, None)
        return extract_image_bytes(cell)

    def _img_feat_from_bytes(self, img_bytes: Optional[bytes]) -> Optional[np.ndarray]:
        """将 cam_high bytes -> 下采样灰度特征，用于快速判断静止帧。"""
        if not img_bytes:
            return None
        pil = bytes_to_pil_image(img_bytes)
        if pil is None:
            return None
        try:
            pil_small = pil.resize((self.static_img_downsample, self.static_img_downsample))
            arr = np.asarray(pil_small, dtype=np.float32)  # RGB
            if arr.ndim == 3 and arr.shape[2] == 3:
                gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
            else:
                gray = arr.astype(np.float32)
            return gray
        except Exception:
            return None

    def is_static_frame(
        self,
        curr_state: np.ndarray,
        curr_img_feat: Optional[np.ndarray],
        prev_state: Optional[np.ndarray],
        prev_img_feat: Optional[np.ndarray],
    ) -> Tuple[bool, Dict[str, Any]]:
        """判定当前帧是否“静止帧”（画面、手臂都几乎没有变化）。"""
        info: Dict[str, Any] = {
            "img_mad": None,
            "state_mad": None,
            "img_threshold": self.static_img_mad_threshold,
            "state_threshold": self.static_state_mad_threshold,
        }

        img_ok = False
        if curr_img_feat is not None and prev_img_feat is not None and curr_img_feat.shape == prev_img_feat.shape:
            img_mad = float(np.mean(np.abs(curr_img_feat - prev_img_feat)))
            info["img_mad"] = img_mad
            img_ok = img_mad < self.static_img_mad_threshold

        state_ok = False
        if curr_state is not None and prev_state is not None and curr_state.size and prev_state.size:
            n = int(min(curr_state.size, prev_state.size))
            state_mad = float(np.mean(np.abs(curr_state[:n] - prev_state[:n])))
            info["state_mad"] = state_mad
            state_ok = state_mad < self.static_state_mad_threshold

        # if (info["img_mad"] is not None) and (info["state_mad"] is not None):
        #     is_static = bool(img_ok and state_ok)
        # elif info["img_mad"] is not None:
        #     is_static = bool(img_ok)
        # elif info["state_mad"] is not None:
        #     is_static = bool(state_ok)
        # else:
        #     is_static = False

        if state_ok or img_ok:
            return True, info

        return False, info

    def run(self) -> bool:
        if not self.test_service_info():
            return False

        # 逐帧评估
        per_frame_dir = os.path.join(self.out_dir, "per_frame_results")

        # 同时做 cam_high 视频（VSCode 友好：优先 ffmpeg(H.264/yuv420p/+faststart)；无 ffmpeg 则回退 OpenCV）
        video_path = os.path.join(self.out_dir, "cam_high_with_frame_idx.mp4")
        video_writer = None
        video_frames_written = 0
        use_ffmpeg = False

        first_img = self._get_cam_high_pil(0)
        if first_img is None:
            print("[WARN] cam_high 第0帧无法解码，视频将跳过。")
        else:
            w, h = first_img.size

            if _ffmpeg_available():
                try:
                    video_writer = FFMPEGVideoWriter(video_path, fps=self.fps, width=w, height=h)
                    use_ffmpeg = True
                    print("[OK] Using ffmpeg for video writing (H.264/yuv420p/+faststart).")
                except Exception as e:
                    video_writer = None
                    use_ffmpeg = False
                    print(f"[WARN] ffmpeg available but failed to start writer: {e}")

            if (video_writer is None) and HAS_CV2:
                # 回退：OpenCV VideoWriter（注意：可能生成 VSCode 不兼容的 mp4v）
                try:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(video_path, fourcc, self.fps, (w, h))
                    use_ffmpeg = False
                    print("[WARN] Falling back to OpenCV VideoWriter (mp4v). VSCode may not preview this video.")
                except Exception as e:
                    video_writer = None
                    print(f"[WARN] OpenCV VideoWriter init failed: {e}")

            if video_writer is None:
                print("[WARN] No suitable video writer; video will be skipped.")
        # 用于静止帧判断的上一帧特征缓存
        prev_state: Optional[np.ndarray] = None
        prev_img_feat: Optional[np.ndarray] = None

        for i in range(self.T):
            curr_state = self._get_state_vec(i)
            curr_img_feat = self._img_feat_from_bytes(self._get_cam_high_bytes(i))

            skipped_infer = False
            static_info: Dict[str, Any] = {}

            if i >= 1:
                is_static, static_info = self.is_static_frame(
                    prev_state=prev_state,
                    prev_img_feat=prev_img_feat,
                    curr_state=curr_state,
                    curr_img_feat=curr_img_feat,
                )
                if is_static:
                    skipped_infer = True
                    ok, infer_json, latency = True, {"success": True, "skipped_infer": True}, 0.0
                else:
                    payload = self.build_payload_for_frame(i)
                    ok, infer_json, latency = self.infer(payload)
            else:
                payload = self.build_payload_for_frame(i)
                ok, infer_json, latency = self.infer(payload)

            # 先更新缓存（用于下一帧的静止判断）
            prev_state = curr_state
            prev_img_feat = curr_img_feat


            self.http_ok.append(ok)
            self.latencies.append(latency)

            # 保存每帧推理结果（便于回溯）
            frame_dump = {
                "frame_idx": i,
                "ok": ok,
                "latency_s": latency,
                "skipped_infer": skipped_infer,
                "static_check": static_info,
                "infer": infer_json,
            }
            with open(os.path.join(per_frame_dir, f"infer_{i:06d}.json"), "w", encoding="utf-8") as f:
                json.dump(frame_dump, f, ensure_ascii=False, indent=2)
            # 误差计算
            if skipped_infer:
                # 静止帧：直接置 0（不做对齐、也不依赖服务端输出）
                L = 0
                self.frame_valid_steps.append(L)
                e_mse, e_rmse = 0.0, 0.0
            else:
                true_seq = self.get_truth_action_seq(i)
                pred_seq = self.extract_pred_seq(infer_json) if ok else np.zeros((0, 0), dtype=np.float32)

                L = int(min(len(true_seq), len(pred_seq)))
                self.frame_valid_steps.append(L)

                if ok and L > 0:
                    true_aligned = true_seq[:L]
                    pred_aligned = pred_seq[:L]

                    # 如果维度不同（例如 pred 多一列/少一列），按最小维度截断
                    D = int(min(true_aligned.shape[1], pred_aligned.shape[1])) if true_aligned.ndim == 2 and pred_aligned.ndim == 2 else 0
                    if D > 0:
                        true_aligned = true_aligned[:, :D]
                        pred_aligned = pred_aligned[:, :D]
                        e_mse = mse(pred_aligned, true_aligned)
                        e_rmse = rmse(pred_aligned, true_aligned)
                    else:
                        e_mse, e_rmse = np.nan, np.nan
                else:
                    e_mse, e_rmse = np.nan, np.nan

            self.frame_errors_mse.append(float(e_mse))
            self.frame_errors_rmse.append(float(e_rmse))


            # 写入视频帧（cam_high）
            if video_writer is not None:
                if self.video_limit is None or video_frames_written < self.video_limit:
                    img = self._get_cam_high_pil(i)
                    if img is not None:
                        img2 = overlay_frame_idx(img, i)
                        arr_rgb = np.array(img2)  # RGB uint8
                        # Convert RGB -> BGR without requiring cv2
                        frame_bgr = arr_rgb[:, :, ::-1]
                        if use_ffmpeg:
                            # ffmpeg writer expects BGR24
                            video_writer.write(frame_bgr)
                        else:
                            # OpenCV VideoWriter expects BGR
                            video_writer.write(frame_bgr)
                        video_frames_written += 1

            if self.sleep_s > 0:
                time.sleep(self.sleep_s)

            # 轻量日志
            if i % 50 == 0 or i == self.T - 1:
                print(f"[{i+1}/{self.T}] ok={ok} L={L} mse={self.frame_errors_mse[-1]:.6f} latency={latency:.3f}s")

        if video_writer is not None:
            try:
                if use_ffmpeg:
                    video_writer.close()
                else:
                    video_writer.release()
                print(f"[OK] video saved: {video_path}")
            except Exception as e:
                print(f"[WARN] video finalize failed: {e}")

        # 汇总保存
        self._save_metrics()
        self._plot_errors()
        self._save_report()

        print("[DONE] framewise evaluation completed.")
        return True

    def _get_cam_high_pil(self, i: int) -> Optional[Image.Image]:
        row = self.df.iloc[i]
        key = f"{self.col_img_prefix}cam_high"
        # key = "image"
        cell = row.get(key, None)
        img_bytes = extract_image_bytes(cell)
        if img_bytes is None:
            return None
        return bytes_to_pil_image(img_bytes)

    def _save_metrics(self) -> None:
        # 保存逐帧误差表
        dfm = pd.DataFrame({
            "frame_idx": np.arange(self.T),
            "ok": self.http_ok,
            "latency_s": self.latencies,
            "valid_steps": self.frame_valid_steps,
            "mse": self.frame_errors_mse,
            "rmse": self.frame_errors_rmse,
        })
        csv_path = os.path.join(self.out_dir, "frame_errors.csv")
        dfm.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"[OK] metrics csv saved: {csv_path}")

    def _plot_errors(self) -> None:
        if not HAS_MPL:
            print("[WARN] matplotlib 不可用，跳过误差曲线绘图。")
            return

        import math

        x = np.arange(self.T)
        y = np.asarray(self.frame_errors_mse, dtype=np.float64)

        # =========================
        # 关键参数（你最关心的）
        # =========================
        x_tick_step = 10          # 时间轴刻度间隔（帧）
        inches_per_100_frames = 2.5  # 每 100 帧占用的图宽（英寸，可微调）

        # 根据 episode 长度自适应 figure 宽度
        fig_width = max(
            10,  # 最小宽度，防止太窄
            # math.ceil(self.T / 100) * inches_per_100_frames
            self.T / 20
        )
        fig_height = 4.5  # 高度一般不需要太大

        plt.figure(figsize=(fig_width, fig_height))
        plt.plot(x, y, linewidth=1.5)

        # x 轴刻度控制
        xticks = np.arange(0, self.T, x_tick_step)
        plt.xticks(xticks)

        plt.xlabel("Frame Index")
        plt.ylabel("MSE (aligned horizon)")
        plt.title("Per-frame Prediction Error (MSE)")
        plt.grid(True, linestyle="--", alpha=0.6)

        # 防止刻度文字被裁掉
        plt.tight_layout()

        fig_path = os.path.join(self.out_dir, "figs", "per_frame_mse.png")
        plt.savefig(fig_path, dpi=200)
        plt.close()

        print(f"[OK] error plot saved: {fig_path}")

    def _save_report(self) -> None:
        y = np.asarray(self.frame_errors_mse, dtype=np.float64)
        ok = np.asarray(self.http_ok, dtype=bool)
        valid = np.asarray(self.frame_valid_steps, dtype=np.int64)

        # 统计时忽略 NaN
        y_valid = y[~np.isnan(y)]
        report = {
            "data_file": self.data_file,
            "url": self.url,
            "total_frames": self.T,
            "action_horizon": self.action_horizon,
            "pred_field": self.pred_field,
            "success_frames": int(ok.sum()),
            "failed_frames": int((~ok).sum()),
            "frames_with_valid_alignment": int(np.sum(valid > 0)),
            "mse_mean": float(np.mean(y_valid)) if y_valid.size else None,
            "mse_std": float(np.std(y_valid)) if y_valid.size else None,
            "mse_min": float(np.min(y_valid)) if y_valid.size else None,
            "mse_max": float(np.max(y_valid)) if y_valid.size else None,
        }

        path = os.path.join(self.out_dir, "test_report.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"[OK] report saved: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:5002")
    ap.add_argument("--data-file", required=True)
    ap.add_argument("--out-dir", default="./out_eval8")
    ap.add_argument("--action-horizon", type=int, default=50)
    ap.add_argument("--pred-field", default="qpos", help="服务端返回里用于取预测动作序列的字段名（默认 qpos）")
    ap.add_argument("--prompt", default="Use the left arm to hold the red-capped bottle from the table")
    ap.add_argument("--timeout-s", type=int, default=100)
    ap.add_argument("--sleep-s", type=float, default=0.0, help="每帧请求后 sleep，用于限速")
    ap.add_argument("--fps", type=int, default=20, help="输出 cam_high 视频帧率")
    ap.add_argument("--video-limit", type=int, default=None, help="只写前 N 帧到视频（调试/省时间）")
    args = ap.parse_args()

    tester = EpisodeFramewiseTester(
        url=args.url,
        data_file=args.data_file,
        out_dir=args.out_dir,
        action_horizon=args.action_horizon,
        pred_field=args.pred_field,
        prompt=args.prompt,
        timeout_s=args.timeout_s,
        sleep_s=args.sleep_s,
        fps=args.fps,
        video_limit=args.video_limit,
    )
    ok = tester.run()
    raise SystemExit(0 if ok else 2)


if __name__ == "__main__":
    main()