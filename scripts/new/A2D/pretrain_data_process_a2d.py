import json
import os
import numpy as np
from tqdm import tqdm
import h5py
from PIL import Image
import logging
from typing import List, Dict, Any, Tuple
from scipy.spatial.transform import Rotation as R
import sys

sys.path.append("/share/project/fengli/code")
from action_token.action_chunk_to_fast_token import ActionChunkProcessor

# 环境变量配置
SAMPLE_INTERVAL = int(os.getenv("SAMPLE_INTERVAL", 1))
DATA_VERSION = os.getenv("DATA_VERSION", "data_test")

# 日志配置
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Tokenizer缓存
_TOKENIZER_CACHE: dict[int, ActionChunkProcessor] = {}


def get_tokenizer(max_len: int) -> ActionChunkProcessor:
    """获取缓存的动作tokenizer"""
    if max_len not in _TOKENIZER_CACHE:
        _TOKENIZER_CACHE[max_len] = ActionChunkProcessor(max_len=max_len)
        logger.debug(f"初始化tokenizer (PID={os.getpid()}, max_len={max_len})")
    return _TOKENIZER_CACHE[max_len]


def numpy_to_python(obj):
    """将numpy类型转换为Python原生类型"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    raise TypeError(f"无法序列化类型: {type(obj)}")


def quat_to_6d(quaternions: np.ndarray) -> np.ndarray:
    """四元数转6D表示"""
    is_single = quaternions.ndim == 1
    if is_single:
        quaternions = quaternions[np.newaxis, :]

    rot = R.from_quat(quaternions)
    matrix = rot.as_matrix()
    d6 = np.concatenate([matrix[..., :, 0], matrix[..., :, 1]], axis=-1)

    return d6.flatten() if is_single else d6


def compute_d6_axis_angle_deltas(d6_sequence: np.ndarray) -> np.ndarray:
    """计算6D姿态序列的axis-angle delta"""
    if d6_sequence.shape[0] < 2:
        raise ValueError("序列长度必须 >= 2")

    # 将6D转换为旋转矩阵
    d6_t = d6_sequence[:-1]
    d6_t_plus_1 = d6_sequence[1:]

    # 重建旋转矩阵
    col1_t = d6_t[:, :3]
    col2_t = d6_t[:, 3:6]
    col3_t = np.cross(col1_t, col2_t)
    R_t = np.stack([col1_t, col2_t, col3_t], axis=-1)

    col1_t1 = d6_t_plus_1[:, :3]
    col2_t1 = d6_t_plus_1[:, 3:6]
    col3_t1 = np.cross(col1_t1, col2_t1)
    R_t_plus_1 = np.stack([col1_t1, col2_t1, col3_t1], axis=-1)

    # 计算相对旋转
    delta_R = R_t_plus_1 @ R_t.transpose(0, 2, 1)

    # 转换为axis-angle
    delta_axis_angle = R.from_matrix(delta_R).as_rotvec()

    return delta_axis_angle


class TrajectoryProcessor:
    """A2D机器人轨迹数据处理器（6D版本）"""

    def __init__(self, args):
        self.args = args
        self.raw_task = args.instruction or "Please follow the task instruction."
        self._load_norm_params()
        self.json_file_writer_320 = None
        self.json_file_writer_640 = None

    def _load_norm_params(self):
        """加载归一化参数"""
        with open(self.args.normal_path, "r") as f:
            params = json.load(f)

        self.action_scale = np.array(params["action.eepose"]["scale_"])
        self.action_offset = np.array(params["action.eepose"]["offset_"])
        self.state_eepose_scale = np.array(params["state.eepose"]["scale_"])
        self.state_eepose_offset = np.array(params["state.eepose"]["offset_"])

    def create_json_entry(
        self,
        task_desc: str,
        image_paths: List[str],
        token_path: str,
        state_eepose_path: str,
        action_eepose_path: str,
    ) -> Dict[str, Any]:
        """生成训练数据JSON条目"""
        return {
            "raw_task": task_desc,
            "image": image_paths,
            "action_token": token_path,
            "state": {"eepose": state_eepose_path},
            "action": {"eepose": action_eepose_path},
            "conversations": [
                {
                    "from": "human",
                    "value": f"You are controlling an A2D dual-arm robot. Your task is to adjust the end effector (EEF) poses at 30Hz to complete a specified task. You need to output control tokens that can be decoded into a 30×14 action sequence. The sequence has 30 consecutive actions, each with 14 dimensions. The first 7 dimensions control the right arm EEF, and the last 7 dimensions control the left arm EEF. Each EEPose here includes 3 delta position(xyz) + 3 delta orientation(axis-angle) + 1 gripper(opening range) Your current visual inputs are: robot head image<image>, robot left wrist image<image> and robot right wrist image<image>. Your overall task is: {task_desc}",
                },
                {"from": "gpt", "value": "<action_token>"},
            ],
        }

    def normalize(
        self, x: np.array, scale: np.array, offset: np.array, clip: bool = True
    ) -> np.array:
        """归一化数据到[-1, 1]"""
        x_norm = x * scale + offset
        if clip:
            np.clip(x_norm, -1, 1, out=x_norm)
        return x_norm

    def denormalize(
        self, x_norm: np.array, scale: np.array, offset: np.array
    ) -> np.array:
        """反归一化"""
        return (np.asarray(x_norm) - offset) / scale

    def process_single_task(self, output_root):
        """处理单个轨迹数据"""
        action_tokenizer = get_tokenizer(self.args.max_len)
        for task in tqdm(self.all_tasks[self.args.start_idx : self.args.end_idx]):
            json_entries_320 = []
            json_entries_640 = []
            try:
                job_id = str(task["job_id"])
                task_id = str(task["task_id"])
                episode_id = str(task["episode_id"])
                sn_code = task.get("sn_code", "")

                path_candidates = []

                if sn_code:
                    path_candidates.append(
                        os.path.join(
                            self.args.data_dir,
                            str(job_id),
                            str(sn_code),
                            str(episode_id),
                        )
                    )

                path_candidates.append(
                    os.path.join(self.args.data_dir, str(job_id), str(episode_id))
                )

                aligned_path = task.get("aligned_data_path")
                if aligned_path:
                    alt = aligned_path
                    if "://" in alt:
                        alt = alt.split("/GENIE/")[-1]
                    alt = alt.strip("/")
                    if alt:
                        path_candidates.append(os.path.join(self.args.data_dir, alt))

                hdf5_path = None
                for candidate in path_candidates:
                    if os.path.exists(os.path.join(candidate, "aligned_joints.h5")):
                        hdf5_path = candidate
                        break

                if hdf5_path is None:
                    logger.error(
                        "未找到 aligned_joints.h5: task=%s job=%s episode=%s",
                        task_id,
                        job_id,
                        episode_id,
                    )
                    self.filtered_traj += 1
                    continue

                cam_path = os.path.join(hdf5_path, "camera")

                label_info = task.get("label_info")
                if (
                    not label_info
                    or "action_config" not in label_info
                    or not label_info["action_config"]
                ):
                    logger.warning(
                        "缺少 action_config，跳过: task=%s job=%s episode=%s",
                        task_id,
                        job_id,
                        episode_id,
                    )
                    self.filtered_traj += 1
                    continue

                with h5py.File(os.path.join(hdf5_path, "aligned_joints.h5"), "r") as f:
                    state_eepose_pos = f["action"]["end"]["position"][:]
                    state_eepose_left_orientation = quat_to_6d(
                        f["action"]["end"]["orientation"][:, 0, :]
                    )
                    state_eepose_right_orientation = quat_to_6d(
                        f["action"]["end"]["orientation"][:, 1, :]
                    )
                    state_eepose_left_gripper = (
                        1 - f["action"]["left_effector"]["position"][:]
                    )
                    state_eepose_right_gripper = (
                        1 - f["action"]["right_effector"]["position"][:]
                    )

                    state_eepose = np.concatenate(
                        [
                            state_eepose_pos[:, 1, :],
                            state_eepose_right_orientation,
                            state_eepose_right_gripper,
                            state_eepose_pos[:, 0, :],
                            state_eepose_left_orientation,
                            state_eepose_left_gripper,
                        ],
                        axis=-1,
                    )

                action_config = label_info["action_config"]
                start = int(action_config[0]["start_frame"])
                end = int(action_config[-1]["end_frame"])

                total_frames = state_eepose.shape[0]
                if total_frames == 0:
                    logger.warning(
                        "轨迹为空，跳过: task=%s job=%s episode=%s",
                        task_id,
                        job_id,
                        episode_id,
                    )
                    self.filtered_traj += 1
                    continue

                loop_start = max(0, min(start, total_frames - 1))
                loop_end = min(max(loop_start, end + 1), total_frames)
                if loop_end - loop_start < 2:
                    logger.warning(
                        "标注范围过短，跳过: task=%s job=%s episode=%s",
                        task_id,
                        job_id,
                        episode_id,
                    )
                    self.filtered_traj += 1
                    continue

                self.total_traj += 1
                uuid = f"{job_id}_{task_id}_{episode_id}"

                images_path_320 = os.path.join(output_root, "images", uuid)
                images_path_640 = os.path.join(output_root, "images_640x480", uuid)
                action_token_path = os.path.join(output_root, "action_token", uuid)

                os.makedirs(images_path_320, exist_ok=True)
                os.makedirs(images_path_640, exist_ok=True)
                os.makedirs(action_token_path, exist_ok=True)

                # 从 label_info.action_config 中提取并拼接所有子任务的 action_text
                task_desc = None
                try:
                    action_config = label_info.get("action_config", [])
                    if action_config and isinstance(action_config, list):
                        # 提取所有非错误的 action_text
                        action_texts = []
                        for action in action_config:
                            if not action.get("is_mistake", False):
                                action_text = action.get("action_text", "").strip()
                                if action_text:
                                    action_texts.append(action_text)

                        # 用 " and " 拼接所有子任务
                        if action_texts:
                            task_desc = " and ".join(action_texts)
                            logger.debug(f"从 action_config 提取任务描述: {task_desc}")
                except Exception as e:
                    logger.warning(f"从 action_config 提取任务描述失败: {e}")

                # 如果 action_config 中没有提取到，尝试从 text 字段提取
                if not task_desc and "text" in task:
                    try:
                        text_data = json.loads(task["text"])
                        if "extra" in text_data and text_data["extra"]:
                            task_desc = text_data["extra"][0]
                        elif "description" in text_data:
                            task_desc = text_data["description"]
                    except Exception as e:
                        logger.warning(f"解析 text 字段失败: {e}")

                # 如果仍然没有，使用命令行传入的 instruction
                if not task_desc:
                    task_desc = (
                        self.raw_task
                        if self.raw_task
                        else "Please follow the task instruction."
                    )

                logger.info(f"处理 {uuid}: {task_desc}")

                # 过滤静止帧
                filtered_action = []
                original_indices = []
                self.original_sample_num += loop_end - loop_start

                filtered_action.append(state_eepose[loop_start])
                original_indices.append(loop_start)

                for i in range(loop_start + 1, loop_end):
                    if not np.allclose(
                        state_eepose[i - 1], state_eepose[i], rtol=1e-5, atol=1e-6
                    ):
                        filtered_action.append(state_eepose[i])
                        original_indices.append(i)

                filtered_action = np.array(filtered_action)
                original_indices = np.array(original_indices)
                self.wo_static_sample_num += len(filtered_action)

                # 检查轨迹长度
                if len(filtered_action) < 15:
                    self.filtered_traj += 1
                    continue

                # 构建训练样本
                construct_num = max(1, len(filtered_action) - self.args.chunk)

                for i in range(construct_num):
                    # 获取动作分块 (31帧)
                    chunk_len = self.args.chunk + 1
                    index = [
                        min(i + j, len(filtered_action) - 1) for j in range(chunk_len)
                    ]
                    action_chunk = filtered_action[index]  # (31, 20)
                    original_index_chunk = original_indices[index]

                    # 计算action delta (30, 14)
                    try:
                        action_delta = np.concatenate(
                            [
                                action_chunk[1:, :3]
                                - action_chunk[:-1, :3],  # 右臂位置delta (30, 3)
                                compute_d6_axis_angle_deltas(
                                    action_chunk[:, 3:9]
                                ),  # 右臂姿态delta (30, 3)
                                action_chunk[1:, [9]],  # 右臂夹爪 (30, 1)
                                action_chunk[1:, 10:13]
                                - action_chunk[:-1, 10:13],  # 左臂位置delta (30, 3)
                                compute_d6_axis_angle_deltas(
                                    action_chunk[:, 13:19]
                                ),  # 左臂姿态delta (30, 3)
                                action_chunk[1:, [19]],  # 左臂夹爪 (30, 1)
                            ],
                            axis=-1,
                        )  # 总维度: 3+3+1 + 3+3+1 = 14
                    except Exception as e:
                        logger.error(f"计算delta失败 {uuid}/{i}: {e}")
                        continue

                    # 归一化action delta
                    nor_action_delta = self.normalize(
                        action_delta, self.action_scale, self.action_offset
                    )

                    # 去掉最后一帧，使state与action_delta帧数一致
                    action_chunk = action_chunk[:-1]  # (30, 20)

                    # 归一化state
                    nor_state = self.normalize(
                        action_chunk, self.state_eepose_scale, self.state_eepose_offset
                    )

                    # 生成文件路径
                    base_idx = original_indices[i]
                    token_file = os.path.join(
                        action_token_path, f"token_{base_idx}.npy"
                    )
                    chunk_file = os.path.join(
                        action_token_path, f"chunk_{base_idx}.npy"
                    )
                    delta_file = os.path.join(
                        action_token_path, f"delta_{base_idx}.npy"
                    )
                    indices_file = os.path.join(
                        action_token_path, f"original_indices_{base_idx}.npy"
                    )

                    # 生成action token
                    action_token = action_tokenizer.process_action_chunk_to_fast_token(
                        nor_action_delta
                    )

                    # 保存action token和npy文件（只保存一次）
                    np.save(token_file, action_token)
                    np.save(chunk_file, nor_state)
                    np.save(delta_file, nor_action_delta)
                    np.save(indices_file, original_index_chunk)

                    # 处理并保存两个尺寸的图像
                    views = ["head_color", "hand_left_color", "hand_right_color"]

                    # 320x240版本
                    image_paths_320 = []
                    frame_dir_320 = os.path.join(images_path_320, str(base_idx))
                    os.makedirs(frame_dir_320, exist_ok=True)

                    for view in views:
                        src = os.path.join(cam_path, str(base_idx), f"{view}.jpg")
                        dst = os.path.join(frame_dir_320, f"{view}.jpg")
                        img = Image.open(src).convert("RGB")
                        img.resize((320, 240)).save(dst)
                        image_paths_320.append(dst)

                    # 640x480版本
                    image_paths_640 = []
                    frame_dir_640 = os.path.join(images_path_640, str(base_idx))
                    os.makedirs(frame_dir_640, exist_ok=True)

                    for view in views:
                        src = os.path.join(cam_path, str(base_idx), f"{view}.jpg")
                        dst = os.path.join(frame_dir_640, f"{view}.jpg")
                        img = Image.open(src).convert("RGB")
                        img.resize((640, 480)).save(dst)
                        image_paths_640.append(dst)

                    # 生成两个JSON条目
                    json_task_320 = self.create_json_entry(
                        task_desc,
                        image_paths_320,
                        token_file,
                        chunk_file,
                        delta_file,
                    )
                    json_entries_320.append(json_task_320)

                    json_task_640 = self.create_json_entry(
                        task_desc,
                        image_paths_640,
                        token_file,
                        chunk_file,
                        delta_file,
                    )
                    json_entries_640.append(json_task_640)
            except Exception as e:
                logger.error(f"处理轨迹失败: {e}")
                continue

            # 采样并写入两个JSONL
            try:
                if json_entries_320 and json_entries_640:
                    sampled_320 = json_entries_320[::SAMPLE_INTERVAL]
                    if json_entries_320[-1] not in sampled_320:
                        sampled_320.append(json_entries_320[-1])

                    sampled_640 = json_entries_640[::SAMPLE_INTERVAL]
                    if json_entries_640[-1] not in sampled_640:
                        sampled_640.append(json_entries_640[-1])

                    for entry in sampled_320:
                        self.json_file_writer_320.write(json.dumps(entry) + "\n")

                    for entry in sampled_640:
                        self.json_file_writer_640.write(json.dumps(entry) + "\n")

                    self.json_file_writer_320.flush()
                    self.json_file_writer_640.flush()
                    logger.debug(
                        f"写入 320x240: {len(sampled_320)} 条, 640x480: {len(sampled_640)} 条"
                    )
            except Exception as e:
                logger.error(f"写入JSONL失败: {e}")

    def process_task(self):
        """处理所有任务"""
        self.original_sample_num = 0
        self.wo_static_sample_num = 0
        self.filtered_traj = 0
        self.total_traj = 0

        # 输出目录：/share/project/fengli/data/{task_name}/a2d_train_data
        task_output_dir = os.path.join(self.args.root_path, self.args.task_name)
        output_root = os.path.join(task_output_dir, "a2d_train_data")
        os.makedirs(output_root, exist_ok=True)

        # JSONL输出在任务目录下
        os.makedirs(task_output_dir, exist_ok=True)

        jsonl_path_320 = os.path.join(
            task_output_dir,
            f"a2d_train_{self.args.task_name}_320x240_{self.args.start_idx}_{self.args.end_idx}.jsonl",
        )
        jsonl_path_640 = os.path.join(
            task_output_dir,
            f"a2d_train_{self.args.task_name}_640x480_{self.args.start_idx}_{self.args.end_idx}.jsonl",
        )

        with open(jsonl_path_320, "w") as self.json_file_writer_320, open(
            jsonl_path_640, "w"
        ) as self.json_file_writer_640:
            self.process_single_task(output_root)

        logger.info(f"处理完成:")
        logger.info(f"  原始帧数: {self.original_sample_num}")
        logger.info(f"  过滤后帧数: {self.wo_static_sample_num}")
        logger.info(f"  总轨迹数: {self.total_traj}")
        logger.info(f"  过滤轨迹数: {self.filtered_traj}")

    def run(self):
        """运行数据处理流程"""
        drop_list = set([1765, 1790, 1830, 1832])

        logger.info(f"扫描数据目录: {self.args.data_dir}")
        logger.info(f"任务类型: {self.args.task_name}")
        logger.info(f"Job ID 范围: {self.args.start_task_id} - {self.args.end_task_id}")

        all_tasks = []
        annotation_cache: Dict[str, Dict[Tuple[str, str], Dict[str, Any]]] = {}

        for job_id_dir in os.listdir(self.args.data_dir):
            try:
                job_id_num = int(job_id_dir)
                if not (self.args.start_task_id <= job_id_num <= self.args.end_task_id):
                    continue

                if job_id_num in drop_list or job_id_dir in drop_list:
                    logger.info(f"跳过 Job ID {job_id_dir} (在过滤列表中)")
                    continue

                job_dir_path = os.path.join(self.args.data_dir, job_id_dir)

                annotation_dict = annotation_cache.get(job_id_dir)
                if annotation_dict is None:
                    annotation_dict = {}
                    annotation_file = os.path.join(job_dir_path, f"{job_id_dir}.json")
                    if os.path.exists(annotation_file):
                        try:
                            with open(annotation_file, "r", encoding="utf-8") as f:
                                annotation_list = json.load(f)
                            for entry in annotation_list:
                                job_key = entry.get("job_id")
                                sn_key = entry.get("sn_code")
                                episode_key = entry.get("episode_id")
                                if episode_key is None:
                                    continue
                                episode_key = str(episode_key)
                                if job_key is not None:
                                    annotation_dict[(str(job_key), episode_key)] = entry
                                if sn_key is not None:
                                    annotation_dict[(str(sn_key), episode_key)] = entry
                        except Exception as e:
                            logger.warning(f"加载标注文件失败 {annotation_file}: {e}")
                    annotation_cache[job_id_dir] = annotation_dict

                for sn_code_dir in os.listdir(job_dir_path):
                    sn_dir = os.path.join(job_dir_path, sn_code_dir)
                    if not os.path.isdir(sn_dir):
                        continue

                    for episode_id in os.listdir(sn_dir):
                        episode_dir = os.path.join(sn_dir, episode_id)
                        if not os.path.isdir(episode_dir):
                            continue

                        meta_file = os.path.join(episode_dir, "meta_info.json")
                        if os.path.exists(meta_file):
                            try:
                                with open(meta_file, "r") as f:
                                    task_data = json.load(f)
                                    episode_id_str = str(episode_id)

                                    job_id_str = str(
                                        task_data.get("job_id", job_id_dir)
                                    )
                                    task_id_str = str(task_data.get("task_id", ""))

                                    if not task_id_str:
                                        logger.warning(
                                            "meta_info.json 缺少 task_id，跳过: job=%s episode=%s",
                                            job_id_str,
                                            episode_id_str,
                                        )
                                        continue

                                    task_data.update(
                                        {
                                            "job_id": job_id_str,
                                            "task_id": task_id_str,
                                            "episode_id": episode_id_str,
                                            "sn_code": sn_code_dir,
                                        }
                                    )

                                    annotation = annotation_dict.get(
                                        (job_id_str, episode_id_str)
                                    )
                                    if not annotation:
                                        sn_key = task_data.get("sn_code")
                                        if sn_key:
                                            annotation = annotation_dict.get(
                                                (str(sn_key), episode_id_str)
                                            )
                                    if not annotation:
                                        logger.warning(
                                            "缺少标注，跳过: task=%s job=%s episode=%s",
                                            task_id_str,
                                            job_id_str,
                                            episode_id_str,
                                        )
                                        continue

                                    label_info = annotation.get("label_info")
                                    if not label_info or not label_info.get(
                                        "action_config"
                                    ):
                                        logger.warning(
                                            "标注缺少 action_config，跳过: task=%s job=%s episode=%s",
                                            task_id_str,
                                            job_id_str,
                                            episode_id_str,
                                        )
                                        continue

                                    task_data["label_info"] = label_info
                                    if "text" in annotation:
                                        task_data["text"] = annotation["text"]
                                    if "aligned_data_path" in annotation:
                                        task_data["aligned_data_path"] = annotation[
                                            "aligned_data_path"
                                        ]
                                    if "sn_code" in annotation:
                                        task_data["sn_code"] = annotation["sn_code"]

                                    all_tasks.append(task_data)
                            except Exception as e:
                                logger.warning(f"读取失败 {meta_file}: {e}")
            except Exception:
                continue

        self.all_tasks = all_tasks

        logger.info(f"共找到 {len(self.all_tasks)} 条有效轨迹")
        self.process_task()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="A2D机器人轨迹数据处理（6D版本）")
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
        default="/share/project/caomingyu/a2d_data",
        help="输入数据目录",
    )
    parser.add_argument(
        "--root_path",
        type=str,
        default="/share/project/fengli/data",
        help="输出根目录",
    )
    parser.add_argument(
        "--normal_path",
        type=str,
        default="/share/project/fengli/code/norm/a2d_norm.json",
        help="归一化参数文件",
    )
    parser.add_argument("--start_task_id", type=int, required=True, help="起始job_id")
    parser.add_argument("--end_task_id", type=int, required=True, help="结束job_id")
    parser.add_argument("--max_len", type=int, default=256, help="tokenizer最大长度")
    parser.add_argument("--chunk", type=int, default=30, help="动作序列长度")
    parser.add_argument(
        "--instruction",
        type=str,
        default=None,
        help="写入 JSONL raw_task 字段的任务指令（可选）。如果不提供，将自动从标注文件的 action_config 中提取并拼接子任务的 action_text",
    )
    parser.add_argument("--start_idx", type=int, default=0, help="起始轨迹索引")
    parser.add_argument("--end_idx", type=int, default=30, help="结束轨迹索引")

    args = parser.parse_args()

    logger.info(f"任务类型: {args.task_name}")
    logger.info(f"开始处理轨迹 {args.start_idx}-{args.end_idx}")
    processor = TrajectoryProcessor(args)
    processor.run()
    logger.info("处理完成")
    print("Finish")

    # 清理tokenizer缓存，释放资源
    _TOKENIZER_CACHE.clear()

    # 强制退出
    import sys
    import os

    os._exit(0)
