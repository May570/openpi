"""
Script to convert Aloha hdf5 data to the LeRobot dataset v2.0 format.

Example usage: python /share/project/wujiling/openpi/scripts/agilex_hdf5_to_lerobot/A2D.py
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
import cv2
import json


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = False
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

REMOTE_PREFIX = "zhiyuan-raw://genie-data-base-zhiyuan-1347244451/GENIE"

TARGET_H, TARGET_W = 480, 640

TASK_MAP = {
    3924: "Pick up the first cup from left to right, place it in the correct position, and then pour the coffee into the cup.",
    3926: "Pick up the second cup from left to right, place it in the correct position, and then pour the coffee into the cup.",
    3928: "Pick up the third cup from left to right, place it in the correct position, and then pour the coffee into the cup.",
    3988: "Open the upper drawer, then pick up the item closest to the apple and put it into the drawer, and finally close the upper drawer.",
    4045: "Open the upper drawer, then pick up the item closest to the apple and put it into the drawer, and finally close the upper drawer.",
}


def read_and_resize(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"读取图片失败: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (TARGET_W, TARGET_H))
    return img


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
        "right_1",  
        "right_2",
        "right_3",  
        "right_4",
        "right_5",
        "right_6",
        "right_7",
        "right_gripper",
        "left_1",
        "left_2",
        "left_3",
        "left_4",
        "left_5",
        "left_6",
        "left_7",
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


def map_raw_to_local(raw_data_path: str, raw_dir: Path) -> Path:
    """
    把 json 里的 data_path 映射成本地 episode 路径。
    """
    if not raw_data_path.startswith(REMOTE_PREFIX):
        raise ValueError(
            f"raw_data_path 前缀不匹配：{raw_data_path}\n"
            f"期望前缀: {REMOTE_PREFIX}"
        )
    
    rel = raw_data_path[len(REMOTE_PREFIX) :].lstrip("/")
    rest_parts =rel.split("/")[2:]

    return raw_dir.joinpath(*rest_parts)


def iter_a2d_json_items(raw_dir: Path) -> list[dict]:
    json_files = list(raw_dir.glob("*.json"))

    if len(json_files) == 0:
        raise FileNotFoundError(f"在 {raw_dir} 目录下找不到任何 json 文件")

    if len(json_files) > 1:
        raise RuntimeError(
            f"{raw_dir} 目录下找到多个 json 文件，请确保只有一个: {json_files}"
        )

    json_path = json_files[0]
    print(f"使用 json 文件: {json_path}")
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return list(data)


def load_raw_A2D_episode(
    data_item: dict,
    raw_dir: Path,
) -> tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor]:
    """
    从一个 A2D episode 的 json item 里：
    - 根据 raw_data_path 找到本地 episode 目录
    - 读取 episode 目录/aligned_joints.h5 里的 state/action joint + effector
    - 根据 label_info 截取 [start_frame, end_frame]
    - 读取 episode 目录/camera/<frame>/head_color.jpg, hand_left_color.jpg, hand_right_color.jpg

    返回:
        imgs_per_cam: dict[camera_name -> (N,H,W,3)]
        state: (N,16) torch.FloatTensor
        action: (N,16) torch.FloatTensor
    """
    raw_data_path = data_item["raw_data_path"]
    episode_root = map_raw_to_local(raw_data_path, raw_dir)
    print(f"在 {episode_root} 下找到当前 episode 元数据")

    h5_path = episode_root / "aligned_joints.h5"
    cam_root = episode_root / "camera"

    if not h5_path.exists():
        raise FileNotFoundError(f"找不到 aligned_joints.h5: {h5_path}")
    if not cam_root.exists():
        raise FileNotFoundError(f"找不到 camera 目录: {cam_root}")

    # 1) 读取 state 和 action 的 joint + effector
    with h5py.File(h5_path, "r") as f:
        state_joint = f["state/joint/position"][:]              # (T_s,14)
        state_left = f["state/left_effector/position"][:]       # (T_s,1)
        state_right = f["state/right_effector/position"][:]     # (T_s,1)

        action_joint = f["action/joint/position"][:]            # (T_a,14)
        action_left = f["action/left_effector/position"][:]     # (T_a,1)
        action_right = f["action/right_effector/position"][:]   # (T_a,1)

    T = min(state_joint.shape[0], action_joint.shape[0])
    state_joint, state_left, state_right = state_joint[:T], state_left[:T], state_right[:T]
    action_joint, action_left, action_right = action_joint[:T], action_left[:T], action_right[:T]

    # 16 维 = 右臂7 + 右爪 + 左臂7 + 左爪
    state_qpos = np.concatenate(
        (state_joint[:, 7:], state_right, state_joint[:, :7], state_left),
        axis=1,
    ).astype("float32")   # (T,16)

    action_qpos = np.concatenate(
        (action_joint[:, 7:], action_right, action_joint[:, :7], action_left),
        axis=1,
    ).astype("float32")   # (T,16)

    # 2) 用 label_info 决定截取帧区间
    label = data_item.get("label_info")
    if not label:
        li_list = data_item.get("label_info_list") or []
        label = li_list[0] if li_list else None

    if label and "action_config" in label and label["action_config"]:
        cfgs = [cfg for cfg in label["action_config"] if not cfg.get("is_mistake", False)]
        if cfgs:
            start = cfgs[0]["start_frame"]
            end   = cfgs[-1]["end_frame"]
        else:
            start, end = 0, T - 1
    else:
        start, end = 0, T - 1

    start = max(0, min(start, T - 1))
    end = max(start, min(end, T - 1))
    end_excl = end + 1

    state_qpos = state_qpos[start:end_excl]
    action_qpos = action_qpos[start:end_excl]

    # 3) 读取三路相机图像（保持 HWC，BGR→RGB）
    head_color: list[np.ndarray] = []
    head_left_color: list[np.ndarray] = []
    head_right_color: list[np.ndarray] = []

    for t in range(start, end_excl):
        frame_dir = cam_root / str(t)
        head_path = frame_dir / "head_color.jpg"
        left_path = frame_dir / "hand_left_color.jpg"
        right_path = frame_dir / "hand_right_color.jpg"

        if not head_path.exists() or not left_path.exists() or not right_path.exists():
            raise FileNotFoundError(f"缺少相机图像: {frame_dir}")

        img_h = read_and_resize(head_path)
        img_l = read_and_resize(left_path)
        img_r = read_and_resize(right_path)

        head_color.append(img_h)
        head_left_color.append(img_l)
        head_right_color.append(img_r)

    imgs_per_cam = {
        "cam_high": np.stack(head_color, axis=0), 
        "cam_left_wrist": np.stack(head_left_color, axis=0),
        "cam_right_wrist": np.stack(head_right_color, axis=0),
    }

    state = torch.from_numpy(state_qpos)
    action = torch.from_numpy(action_qpos)

    return imgs_per_cam, state, action


# ========================
# 填充 LeRobotDataset
# ========================

def populate_dataset(
    dataset: LeRobotDataset,
    raw_dir: Path,
    task: str,
    episodes: list[int] | None = None,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    skip_episodes: int = 0,
) -> LeRobotDataset:
    all_items = iter_a2d_json_items(raw_dir)
    print(f"在 {raw_dir} 下共找到 {len(all_items)} 条 episode 元数据")

    converted_count = 0

    for item in tqdm.tqdm(all_items):
        # 只用审核通过的
        status = item.get("episode_status")
        if status not in (None, "approved", "APPROVED"):
            continue

        if converted_count < skip_episodes:
            converted_count += 1
            continue

        imgs_per_cam, state, action = load_raw_A2D_episode(item, raw_dir)
        velocity = None
        effort = None

        # 根据配置决定是否过滤静止帧
        if dataset_config.filter_static_frames:
            filtered_state, filtered_action, filtered_images, filtered_velocity, filtered_effort = filter_static_frames(
                state.numpy(),
                action.numpy(),
                imgs_per_cam,
                velocity.numpy() if velocity is not None else None,
                effort.numpy() if effort is not None else None,
                position_threshold=dataset_config.position_threshold,
                action_threshold=dataset_config.action_threshold,
                min_frames=dataset_config.min_frames,
            )

            state = torch.from_numpy(filtered_state)
            action = torch.from_numpy(filtered_action)
            imgs_per_cam = filtered_images
            velocity = torch.from_numpy(filtered_velocity) if filtered_velocity is not None else None
            effort = torch.from_numpy(filtered_effort) if filtered_effort is not None else None

        num_frames = state.shape[0]
        if num_frames == 0:
            print(f"  episode_id={item.get('episode_id')} 过滤后无帧，跳过")
            continue

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
        converted_count += 1

    return dataset


# ========================
# 顶层入口：port_A2D
# ========================

def port_A2D(
    raw_dir: Path,
    repo_id: str,
    task: str = "DEBUG",
    *,
    start: int | None = None,
    end: int | None = None,
    dir_ids: list[int] | None = None,
    episodes: list[int] | None = None,
    resume: bool = False,
    mode: Literal["video", "image"] = "image",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    """
    raw_dir: 比如 /share/project/caomingyu/a2d_data
             遍历 raw_dir/start 到 end, 或者raw_dir/[dir_ids]，找 aligned_data.json，
             里面的 data_path 再映射到 raw_dir 下面真实 episode 路径。
    """
    if not resume:
        if (LEROBOT_HOME / repo_id).exists():
            shutil.rmtree(LEROBOT_HOME / repo_id)
            print(f"删除已有数据集: {LEROBOT_HOME / repo_id}")

        dataset = create_empty_dataset(
            repo_id,
            robot_type="A2D",
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

    dir_list: list[Path] = []

    if dir_ids:
        for idx in dir_ids:
            d = raw_dir / str(idx)
            if d.exists():
                dir_list.append(d)
            else:
                print(f"目录不存在，跳过: {d}")
    else:
        for idx in range(start, end + 1):
            d = raw_dir / str(idx)
            if d.exists():
                dir_list.append(d)
            else:
                print(f"目录不存在，跳过: {d}")

    for d in dir_list:
        dir_id = int(d.name)
        this_task = task
        if dir_id in TASK_MAP:
            this_task = TASK_MAP[dir_id]
        print(f"[port_A2D] 开始处理目录: {d}")
        dataset = populate_dataset(
            dataset,
            d,
            task=this_task,
            episodes=episodes,
            dataset_config=dataset_config,
            skip_episodes=60,
        )

    return dataset


if __name__ == "__main__":
    import sys
    
    tyro.cli(port_A2D)