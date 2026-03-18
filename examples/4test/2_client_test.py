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
# python examples/4test/2_client_test.py --data-file /share/project/lyx/RoboTwin/lerobot_data/huggingface/lerobot/open_laptop-demo_clean-50/data/chunk-000/episode_000001.parquet

import os
import json
import time
import base64
import argparse
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
        use_images: Tuple[str, ...] = ("cam_high", "cam_left_wrist", "cam_right_wrist"),
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
        self.col_img_prefix = "observation.images."

        # 结果容器
        self.frame_errors_mse: List[float] = []
        self.frame_errors_rmse: List[float] = []
        self.frame_valid_steps: List[int] = []
        self.http_ok: List[bool] = []
        self.latencies: List[Optional[float]] = []

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

    def run(self) -> bool:
        if not self.test_service_info():
            return False

        # 逐帧评估
        per_frame_dir = os.path.join(self.out_dir, "per_frame_results")

        # 同时做 cam_high 视频（先缓存帧，或边推边写）
        video_path = os.path.join(self.out_dir, "cam_high_with_frame_idx.mp4")
        video_writer = None
        video_frames_written = 0

        # 如果用 imageio：我们可以先收集帧数组再写（内存可能大），更稳是边写边 append
        # imageio.v3 的 iio.imwrite 对视频是一次性写；这里用 cv2 边写更稳
        if HAS_CV2:
            # 初始化 writer 需要知道分辨率：先用第0帧 cam_high 解码
            first_img = self._get_cam_high_pil(0)
            if first_img is not None:
                w, h = first_img.size
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(video_path, fourcc, self.fps, (w, h))
            else:
                print("[WARN] cam_high 第0帧无法解码，视频将跳过。")
        else:
            print("[WARN] 未安装 opencv-python，视频将跳过（建议 pip install opencv-python）。")

        for i in range(self.T):
            payload = self.build_payload_for_frame(i)
            ok, infer_json, latency = self.infer(payload)

            self.http_ok.append(ok)
            self.latencies.append(latency)

            # 保存每帧推理结果（便于回溯）
            frame_dump = {
                "frame_idx": i,
                "ok": ok,
                "latency_s": latency,
                "infer": infer_json,
            }
            with open(os.path.join(per_frame_dir, f"infer_{i:06d}.json"), "w", encoding="utf-8") as f:
                json.dump(frame_dump, f, ensure_ascii=False, indent=2)

            # 误差计算
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
                        # PIL RGB -> BGR for cv2
                        frame_bgr = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2BGR)
                        video_writer.write(frame_bgr)
                        video_frames_written += 1

            if self.sleep_s > 0:
                time.sleep(self.sleep_s)

            # 轻量日志
            if i % 50 == 0 or i == self.T - 1:
                print(f"[{i+1}/{self.T}] ok={ok} L={L} mse={self.frame_errors_mse[-1]:.6f} latency={latency:.3f}s")

        if video_writer is not None:
            video_writer.release()
            print(f"[OK] video saved: {video_path}")

        # 汇总保存
        self._save_metrics()
        self._plot_errors()
        self._save_report()

        print("[DONE] framewise evaluation completed.")
        return True

    def _get_cam_high_pil(self, i: int) -> Optional[Image.Image]:
        row = self.df.iloc[i]
        key = f"{self.col_img_prefix}cam_high"
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

        x = np.arange(self.T)
        y = np.asarray(self.frame_errors_mse, dtype=np.float64)

        plt.figure()
        plt.plot(x, y)
        plt.xlabel("Frame Index")
        plt.ylabel("MSE (aligned horizon)")
        plt.title("Per-frame Prediction Error (MSE)")
        plt.grid(True)

        fig_path = os.path.join(self.out_dir, "figs", "per_frame_mse.png")
        plt.savefig(fig_path, dpi=200, bbox_inches="tight")
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
    ap.add_argument("--out-dir", default="./out_eval")
    ap.add_argument("--action-horizon", type=int, default=50)
    ap.add_argument("--pred-field", default="qpos", help="服务端返回里用于取预测动作序列的字段名（默认 qpos）")
    ap.add_argument("--prompt", default="pick up the orange and put it into the basket")
    ap.add_argument("--timeout-s", type=int, default=100)
    ap.add_argument("--sleep-s", type=float, default=0.0, help="每帧请求后 sleep，用于限速")
    ap.add_argument("--fps", type=int, default=30, help="输出 cam_high 视频帧率")
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
