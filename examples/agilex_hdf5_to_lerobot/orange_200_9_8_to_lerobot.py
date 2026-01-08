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

PROMPT_MAP = {
    "task_Songling5_eval50_task1": "Pick up the pencil sharpener and place it to the left of the stapler.",
    "task_Songling5_eval50_task2": "Pick up the bread and place it into the basket.",
    "task_Songling5_eval50_task3": "Pick up the pot and place it into the plate on the right side of the table.",
    "task_Songling5_eval50_task4": "Pick up the fruit and place it into the bowl.",
    "task_Songling5_eval50_task5": "Pick up the object closest to the toothpaste and place it into the basket.",
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
    imgs_per_cam = {}
    for camera in cameras:
        uncompressed = ep[f"/observations/images/{camera}"].ndim == 4

        if uncompressed:
            # load all images in RAM
            imgs_array = ep[f"/observations/images/{camera}"][:]
        else:
            import cv2

            # load one compressed image after the other in RAM and uncompress
            imgs_array = []
            for data in ep[f"/observations/images/{camera}"]:
                imgs_array.append(cv2.cvtColor(cv2.imdecode(data, 1), cv2.COLOR_BGR2RGB))
            imgs_array = np.array(imgs_array)

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
        
        imgs_per_cam, state, action, velocity, effort = load_raw_episode_data(ep_path)
        
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
            
            # 使用qpos_2_joint_positions转换过滤后的state和action
            state = torch.from_numpy(filtered_state)
            action = torch.from_numpy(filtered_action)
            
            # 更新其他数据
            imgs_per_cam = filtered_images
            velocity = torch.from_numpy(filtered_velocity) if filtered_velocity is not None else None
            effort = torch.from_numpy(filtered_effort) if filtered_effort is not None else None
        else:
            # 不过滤时，直接转换
            state = torch.from_numpy(state.numpy())
            action = torch.from_numpy(action.numpy())
        
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

    # if not raw_dir.exists():
    #     if raw_repo_id is None:
    #         raise ValueError("raw_repo_id must be provided if raw_dir does not exist")
    #     download_raw(raw_dir, repo_id=raw_repo_id)

    hdf5_files = sorted(raw_dir.rglob("episode_*.hdf5"))

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
    dataset.consolidate()

    if push_to_hub:
        dataset.push_to_hub()


def port_multiple_tasks(
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
    批量处理多个任务文件夹，将它们合并到一个LeRobot数据集中
    """
    if (LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)  # 删除整个目录树(所有文件及子目录)

    # 找到所有任务文件夹 (例如 task_Songling5_eval50_task1, task_Songling5_eval50_task2 ...)
    task_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    print(f"Found {len(task_dirs)} task directories: {[d.name for d in task_dirs]}")

    if not task_dirs:
        raise ValueError(f"No task directories found in {base_dir}")

    results = {}

    for task_dir in task_dirs:
        task_name = task_dir.name
        repo_task_id = f"{repo_id}/{task_name}"  # 每个任务一个独立的数据集

        # 查找这个任务文件夹里的所有 episode hdf5 文件 (递归包含日期子目录)
        hdf5_files = sorted(task_dir.rglob("episode_*.hdf5"))
        if not hdf5_files:
            print(f"⚠️ Warning: No HDF5 files found in {task_dir}")
            continue

        print(f"\nProcessing task: {task_name} ({len(hdf5_files)} episodes)")

        # 为每个任务新建数据集
        dataset = create_empty_dataset(
            repo_task_id,
            robot_type="mobile_aloha" if is_mobile else "aloha",
            mode=mode,
            has_effort=has_effort(hdf5_files),
            has_velocity=has_velocity(hdf5_files),
            dataset_config=dataset_config,
        )

        # 填充数据
        prompt = PROMPT_MAP.get(task_name, task_name)
        dataset = populate_dataset(
            dataset,
            hdf5_files,
            task=prompt,   # 可以替换成标准化 prompt
            episodes=episodes,
        )
    
    # dataset.consolidate()
    # print(f"\nDataset consolidated with {len(dataset)} episodes")
    
    # if push_to_hub:
    #     dataset.push_to_hub()
    #     print("Dataset pushed to hub successfully")
    
    results[task_name] = dataset

    return results


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
    
    if len(sys.argv) > 1 and sys.argv[1] == "multiple":
        # 使用批量处理模式
        tyro.cli(port_multiple_tasks)
    else:
        # 使用单任务处理模式
        tyro.cli(port_aloha)