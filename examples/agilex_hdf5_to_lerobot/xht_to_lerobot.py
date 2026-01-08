"""
Script to convert Aloha hdf5 data to the LeRobot dataset v2.0 format.

Example usage: uv run examples/aloha_real/convert_aloha_data_to_lerobot.py --raw-dir /path/to/raw/data --repo-id <org>/<dataset-name>
"""

import dataclasses
from pathlib import Path
import shutil
from typing import Literal

import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME as LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import torch
import tqdm
import tyro
import csv
import cv2

PROMPT_MAP = {
    "白色短袖": "Pick up the pencil sharpener and place it to the left of the stapler.",
    "儿童牛仔短裤": "Pick up the bread and place it into the basket.",
    "黑色短袖": "Pick up the pot and place it into the plate on the right side of the table.",
    "卡其色牛仔裤": "Pick up the fruit and place it into the bowl.",
}

@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None
    # 静止帧过滤参数
    filter_static_frames: bool = True
    position_threshold: float = 1e-4
    action_threshold: float = 1e-4
    min_frames: int = 20


DEFAULT_DATASET_CONFIG = DatasetConfig()


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "video",
    *,
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    motors = [
        "right_waist",  # 腰
        "right_shoulder",
        "right_elbow",  # 手肘
        "right_forearm_roll",  # 前臂滚动
        "right_wrist_angle",
        "right_wrist_rotate",
        "right_gripper",
        "left_waist",
        "left_shoulder",
        "left_elbow",
        "left_forearm_roll",
        "left_wrist_angle",
        "left_wrist_rotate",
        "left_gripper",
    ]
    cameras = [
        "cam_high",
        "cam_left_wrist",
        "cam_right_wrist",
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
            "shape": (3, 480, 640),
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
        fps=50,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def get_cameras(hdf5_files: list[Path]) -> list[str]:
    with h5py.File(hdf5_files[0], "r") as ep:
        # ignore depth channel, not currently handled
        return [key for key in ep["/observations/images"].keys() if "depth" not in key]  # noqa: SIM118


def has_velocity(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/qvel" in ep


def has_effort(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/effort" in ep


def load_raw_images_per_camera(ep: h5py.File, cameras: list[str]) -> dict[str, np.ndarray]:
    import numpy as np
    import cv2

    imgs_per_cam = {}
    for camera in cameras:
        dset = ep[f"/observations/images/{camera}"]
        uncompressed = dset.ndim == 4

        # ---------- Uncompressed images ----------
        if uncompressed:
            imgs_array = dset[:]  # (T, H, W, C)
            imgs_per_cam[camera] = imgs_array
            continue

        # ---------- Compressed images ----------
        imgs_list: list[np.ndarray] = []

        for idx in range(len(dset)):
            data = dset[idx]

            # to uint8 buffer
            arr = np.asarray(data, dtype=np.uint8).reshape(-1)

            # condition A: buffer empty
            bad = (arr.size == 0)

            if not bad:
                img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                # condition B: imdecode failed
                bad = (img_bgr is None)
            else:
                img_bgr = None

            if bad:
                print(f"[WARN] Bad frame in camera '{camera}', frame index {idx}. Using previous frame.")

                # ----- Case 1: not first frame → copy previous -----
                if len(imgs_list) > 0:
                    imgs_list.append(imgs_list[-1].copy())
                    continue

                # ----- Case 2: first frame坏了 → 创建一张黑图 -----
                # 尝试读取后面正常的一帧推断尺寸
                h, w = 480, 640  # fallback

                # 自动推断，找到下一帧可以解码的图
                for j in range(idx + 1, len(dset)):
                    arr2 = np.asarray(dset[j], dtype=np.uint8).reshape(-1)
                    if arr2.size == 0:
                        continue
                    img2 = cv2.imdecode(arr2, cv2.IMREAD_COLOR)
                    if img2 is not None:
                        h, w = img2.shape[:2]
                        break

                black = np.zeros((h, w, 3), dtype=np.uint8)
                imgs_list.append(black)
                continue

            # normal frame
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            imgs_list.append(img_rgb)

        # stack back
        if len(imgs_list) == 0:
            raise ValueError(f"No valid images in camera '{camera}'.")

        imgs_array = np.stack(imgs_list, axis=0)
        imgs_per_cam[camera] = imgs_array

    return imgs_per_cam


def qpos_2_joint_positions(qpos:np.ndarray):
        l_joint_pos = qpos[:, 50:56]
        r_joint_pos = qpos[:, 0:6]
        l_gripper_pos = np.array([qpos[:,60]]).reshape(-1,1)
        r_gripper_pos = np.array([qpos[:,10]]).reshape(-1,1)

        l_pos = np.concatenate((l_joint_pos,l_gripper_pos), axis=1)
        r_pos = np.concatenate((r_joint_pos,r_gripper_pos), axis=1)

        return np.concatenate((r_pos,l_pos), axis=1)


def filter_static_frames(
    qpos: np.ndarray, 
    action: np.ndarray, 
    images: dict[str, np.ndarray],
    velocity: np.ndarray | None = None,
    effort: np.ndarray | None = None,
    position_threshold: float = 1e-4,
    action_threshold: float = 1e-4,
    min_frames: int = 10
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], np.ndarray | None, np.ndarray | None]:
    """
    过滤掉起始和结束的静止帧
    
    Args:
        qpos: 关节位置数据 (frames, joints)
        action: 动作数据 (frames, joints)  
        images: 图像数据字典
        velocity: 关节速度数据 (可选)
        effort: 关节力矩数据 (可选)
        position_threshold: 位置变化阈值，用于判断是否静止
        action_threshold: 动作变化阈值，用于判断是否静止
        min_frames: 保留的最小帧数
        
    Returns:
        过滤后的数据元组
    """
    if qpos.shape[0] <= min_frames:
        return qpos, action, images, velocity, effort
    
    # 计算关节位置的变化
    position_diff = np.diff(qpos, axis=0)
    position_magnitude = np.linalg.norm(position_diff, axis=1)
    
    # 计算动作的变化
    action_diff = np.diff(action, axis=0)
    action_magnitude = np.linalg.norm(action_diff, axis=1)
    
    # 找到非静止帧的索引
    non_static_mask = (position_magnitude > position_threshold) | (action_magnitude > action_threshold)
    
    # 找到第一个和最后一个非静止帧
    if np.any(non_static_mask):
        first_non_static = np.where(non_static_mask)[0][0]
        last_non_static = np.where(non_static_mask)[0][-1] + 1
        
        # 确保至少保留min_frames帧
        if last_non_static - first_non_static < min_frames:
            # 如果非静止帧太少，扩展范围
            extra_frames = min_frames - (last_non_static - first_non_static)
            first_non_static = max(0, first_non_static - extra_frames // 2)
            last_non_static = min(qpos.shape[0], last_non_static + extra_frames // 2)
    else:
        # 如果没有检测到运动，保留中间部分
        mid_point = qpos.shape[0] // 2
        half_min = min_frames // 2
        first_non_static = max(0, mid_point - half_min)
        last_non_static = min(qpos.shape[0], mid_point + half_min)
    
    # 应用过滤
    filtered_qpos = qpos[first_non_static:last_non_static]
    filtered_action = action[first_non_static:last_non_static]
    
    filtered_images = {}
    for cam_name, img_array in images.items():
        filtered_images[cam_name] = img_array[first_non_static:last_non_static]
    
    filtered_velocity = None
    if velocity is not None:
        filtered_velocity = velocity[first_non_static:last_non_static]
    
    filtered_effort = None
    if effort is not None:
        filtered_effort = effort[first_non_static:last_non_static]
    
    print(f"  过滤静止帧: {qpos.shape[0]} -> {filtered_qpos.shape[0]} 帧 "
          f"(移除前{first_non_static}帧和后{qpos.shape[0]-last_non_static}帧)")
    
    return filtered_qpos, filtered_action, filtered_images, filtered_velocity, filtered_effort


def load_raw_episode_data(
    ep_path: Path,
) -> tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    with h5py.File(ep_path, "r") as ep:
        state = torch.from_numpy(qpos_2_joint_positions(ep["/observations/qpos"][:]))
        action = torch.from_numpy(qpos_2_joint_positions(ep["/action"][:]))

        velocity = None
        if "/observations/qvel" in ep:
            velocity = torch.from_numpy(ep["/observations/qvel"][:])

        effort = None
        if "/observations/effort" in ep:
            effort = torch.from_numpy(ep["/observations/effort"][:])

        imgs_per_cam = load_raw_images_per_camera(
            ep,
            [
                "cam_high",
                "cam_left_wrist",
                "cam_right_wrist",
            ],
        )

    return imgs_per_cam, state, action, velocity, effort

def load_raw_XHT3_data(
    ep_path: Path,
) -> tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    with h5py.File(ep_path, "r") as ep:
        # 1) 读关节位置, 取前 6 维关节角
        l_pos = ep["/action/qpos/left"][:]   # (T, 6)
        r_pos = ep["/action/qpos/right"][:]   # (T, 6)
        l_grip = ep["/action/gripper/left"][:]   # (T, 1)
        r_grip = ep["/action/gripper/right"][:]  # (T, 1)

        # 关节角 + gripper，组成每臂 7 维
        l_vec = np.concatenate([l_pos, l_grip], axis=-1)  # (T, 7)
        r_vec = np.concatenate([r_pos, r_grip], axis=-1)  # (T, 7)

        # motors 顺序遵守：右在前，左在后
        qpos_14 = np.concatenate([r_vec, l_vec], axis=-1).astype("float32")  # (T, 14)

        # 2) state / action 内容相同
        state = torch.from_numpy(qpos_14)
        action = state.clone()

        velocity = None
        effort = None

        imgs_per_cam = load_raw_images_per_camera(
            ep,
            [
                "cam_high",
                "cam_left_wrist",
                "cam_right_wrist",
            ],
        )

    return imgs_per_cam, state, action, velocity, effort


def populate_dataset(
    dataset: LeRobotDataset,
    hdf5_files: list[Path],
    task: str,
    episodes: list[int] | None = None,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    if episodes is None:
        episodes = range(len(hdf5_files))

    for ep_idx in tqdm.tqdm(episodes):
        ep_path = hdf5_files[ep_idx]
        
        imgs_per_cam, state, action, velocity, effort = load_raw_XHT3_data(ep_path)
        
        # 根据配置决定是否过滤静止帧
        if dataset_config.filter_static_frames:
            filtered_state, filtered_action, filtered_images, filtered_velocity, filtered_effort = filter_static_frames(
                state.numpy(), action.numpy(), imgs_per_cam, 
                velocity.numpy() if velocity is not None else None,
                effort.numpy() if effort is not None else None,
                position_threshold=dataset_config.position_threshold,
                action_threshold=dataset_config.action_threshold,
                min_frames=dataset_config.min_frames
            )
            
            state = filtered_state
            action = filtered_action
            imgs_per_cam = filtered_images
            velocity = torch.from_numpy(filtered_velocity) if filtered_velocity is not None else None
            effort = torch.from_numpy(filtered_effort) if filtered_effort is not None else None
        else:
            state = state.numpy()
            action = action.numpy()

        state = torch.from_numpy(state)
        action = torch.from_numpy(action)
        
        num_frames = state.shape[0]

        for i in range(num_frames):
            frame = {
                "observation.state": state[i],
                "action": action[i],
                "task": task,  
            }

            for camera, img_array in imgs_per_cam.items():
                frame[f"observation.images.{camera}"] = img_array[i]

            if velocity is not None:
                frame["observation.velocity"] = velocity[i]
            if effort is not None:
                frame["observation.effort"] = effort[i]

            dataset.add_frame(frame)

        dataset.save_episode()

    return dataset


def port_aloha(
    raw_dir: Path,
    repo_id: str,
    raw_repo_id: str | None = None,
    task: str = "DEBUG",
    *,
    episodes: list[int] | None = None,
    push_to_hub: bool = True,
    is_mobile: bool = False,
    mode: Literal["video", "image"] = "image",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    if (LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)

    hdf5_files = sorted(raw_dir.rglob("*.h5"))

    dataset = create_empty_dataset(
        repo_id,
        robot_type="mobile_aloha" if is_mobile else "aloha",
        mode=mode,
        has_effort=has_effort(hdf5_files),
        has_velocity=has_velocity(hdf5_files),
        dataset_config=dataset_config,
    )
    dataset = populate_dataset(
        dataset,
        hdf5_files,
        task=task,
        episodes=episodes,
        dataset_config=dataset_config,
    )

    # dataset.consolidate()

    # if push_to_hub:
    #     dataset.push_to_hub()

    return dataset


def print_dataset_info(
    base_dir: Path,
    repo_id: str,
    *,
    episodes: list[int] | None = None,
    push_to_hub: bool = True,
    is_mobile: bool = False,
    mode: Literal["video", "image"] = "image",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    """
    """

    # 查找所有任务文件夹
    task_dirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("task_orange")]
    print(f"Found {len(task_dirs)} task directories: {[d.name for d in task_dirs]}")
    
    if not task_dirs:
        raise ValueError(f"No task directories found in {base_dir}")
    
    # 检查第一个任务文件夹来获取数据特征(所有任务文件夹基本格式一致)
    first_task_dir = task_dirs[0]
    hdf5_files = sorted(first_task_dir.glob("episode_*.hdf5"))
    
    if not hdf5_files:
        raise ValueError(f"No HDF5 files found in {first_task_dir}")
    
    # 创建数据集
    dataset = create_empty_dataset(
        repo_id,
        robot_type="mobile_aloha" if is_mobile else "aloha",
        mode=mode,
        has_effort=has_effort(hdf5_files),
        has_velocity=has_velocity(hdf5_files),
        dataset_config=dataset_config,
    )
    
    print(dataset)
    
    return dataset


if __name__ == "__main__":
    import sys
    
    tyro.cli(port_aloha)