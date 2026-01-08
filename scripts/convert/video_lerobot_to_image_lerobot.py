#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import shutil
import dataclasses
import tqdm
import tyro
import json
import torch
import av
from typing import Literal, Optional, List
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME as LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = False
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "image",
    *,
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    motors = [
        "x",  
        "y",
        "z",  
        "roll",
        "pitch",
        "yaw",
        "gripper",
    ]
    cameras = [
        "cam_high",
    ]

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
    }

    if has_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    if has_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, 256, 256),
            "names": [
                "channels",
                "height",
                "width",
            ],
        }

    if Path(LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)

    # Create a LeRobot Dataset from scratch in oreder to record data.
    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=5,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def load_src_info(raw_dir: Path) -> dict:
    info_path = raw_dir / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"missing meta/info.json: {info_path}")
    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    total_episodes = int(info["total_episodes"])
    chunks_size = int(info["chunks_size"])
    data_tpl = info["data_path"]
    video_tpl = info["video_path"]

    episodes = []
    for ep in range(total_episodes):
        episode_chunk = ep // chunks_size

        parquet_path = raw_dir / data_tpl.format(
            episode_chunk=episode_chunk,
            episode_index=ep,
        )

        video_paths = raw_dir / video_tpl.format(
            episode_chunk=episode_chunk,
            video_key="observation.images.image_0",
            episode_index=ep,
        )
    
        episodes.append({
            "episode_index": ep,
            "parquet_path": parquet_path,
            "video_path": video_paths,
        })

    return episodes


def load_task_table(root: Path) -> list[str]:
    """
    在 root/meta/tasks.jsonl 中读取任务列表，返回索引→任务字符串的表。
    """
    tasks_file = root / "meta" / "tasks.jsonl"
    if not tasks_file.exists():
        raise FileNotFoundError(f"找不到 tasks.jsonl: {tasks_file}")

    table: list[str] = []
    with tasks_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = __import__("json").loads(line)
            # 按 task_index 顺序插入，缺位自动扩容
            idx = obj["task_index"]
            task = obj["task"]
            if idx >= len(table):
                table.extend([""] * (idx - len(table) + 1))
            table[idx] = task
    return table


def decode_video_pyav(mp4_path: Path) -> List[np.ndarray]:
    frames: List[np.ndarray] = []
    container = av.open(str(mp4_path))
    stream = container.streams.video[0]

    for frame in container.decode(stream):
        frames.append(frame.to_ndarray(format="rgb24"))

    container.close()
    return frames


def load_raw_episode(
    data_item: dict,
) -> tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor, int]:
    parquet_path = data_item["parquet_path"]
    video_path = data_item["video_path"]

    if not parquet_path.exists():
        raise FileNotFoundError(f"找不到 parquet: {parquet_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"找不到 video 目录: {video_path}")
    
    # ---- 1) 读 parquet ----
    cols = [
        "observation.state",
        "action",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    ]
    table = pq.read_table(parquet_path, columns=cols)
    df = table.to_pandas()
    T = len(df)

    state = torch.tensor(
        np.stack([np.asarray(x, dtype=np.float32) for x in df["observation.state"]]),
        dtype=torch.float32,
    )
    state = torch.cat([state[:, :6], state[:, 7:]], dim=1)
    action = torch.tensor(
        np.stack([np.asarray(x, dtype=np.float32) for x in df["action"]]),
        dtype=torch.float32,
    )
    task_index = int(df["task_index"].iloc[0])

    # ---- 2) 读 video（只读需要的相机）----
    imgs_per_cam = {}
    frames = decode_video_pyav(video_path)
    if len(frames) != T:
        raise RuntimeError(f"视频帧数不匹配！{video_path} 实际 {len(frames)} 帧，但期望 {T} 帧")
    imgs_per_cam = {"cam_high": np.stack(frames, axis=0)}

    return imgs_per_cam, state, action, task_index


# ========================
# 填充 LeRobotDataset
# ========================

def populate_dataset(
    dataset: LeRobotDataset,
    raw_dir: Path,
    skip_episodes: int = 0,
) -> LeRobotDataset:
    all_items = load_src_info(raw_dir)
    task_table = load_task_table(raw_dir)
    print(f"共 {len(all_items)} 条 episode 元数据")

    converted_count = 0

    for item in tqdm.tqdm(all_items):
        if converted_count < skip_episodes:
            converted_count += 1
            continue

        imgs_per_cam, state, action, task_index = load_raw_episode(item)

        num_frames = state.shape[0]
        for i in range(num_frames):
            frame = {
                "observation.images.cam_high": imgs_per_cam["cam_high"][i],
                "observation.state": state[i],
                "action": action[i],
                "task": task_table[task_index],
            }

            dataset.add_frame(frame)

        dataset.save_episode()
        converted_count += 1

    return dataset


# ========================
# 顶层入口：port_image
# ========================

def port_image(
    raw_dir: Path,
    repo_id: str,
    *,
    start: int | None = None,
    end: int | None = None,
    resume: bool = False,
    mode: Literal["video", "image"] = "image",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    if not resume:
        if (LEROBOT_HOME / repo_id).exists():
            shutil.rmtree(LEROBOT_HOME / repo_id)
            print(f"删除已有数据集: {LEROBOT_HOME / repo_id}")

        dataset = create_empty_dataset(
            repo_id,
            robot_type="widowx",
            mode=mode,
            has_effort=False,
            has_velocity=False,
            dataset_config=dataset_config,
        )
    else:
        print(f"追加模式: 打开已有数据集 {repo_id}")
        dataset = LeRobotDataset(
            repo_id=repo_id,
            tolerance_s=dataset_config.tolerance_s,
            download_videos=False,
        )
        dataset.start_image_writer(
            dataset_config.image_writer_processes,
            dataset_config.image_writer_threads,
        )

    print(f"[port_image] 开始处理目录: {raw_dir}")
    dataset = populate_dataset(
        dataset,
        raw_dir,
        skip_episodes=0,
    )

    return dataset


if __name__ == "__main__":
    tyro.cli(port_image)
