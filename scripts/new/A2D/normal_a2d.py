import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import json
import h5py
import os
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R


class AdvancedQuantileNormalizer(BaseEstimator, TransformerMixin):
    def __init__(
        self, lower_quantile=0.01, upper_quantile=0.99, target_range=(-1, 1), clip=True
    ):
        """
        增强版分位数归一化器

        参数:
            lower_quantile: 下分位数(默认1%)
            upper_quantile: 上分位数(默认99%)
            target_range: 目标范围元组(默认[-1, 1])
            clip: 是否将超出范围的值裁剪到边界(默认True)
        """
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.target_min, self.target_max = target_range
        self.clip = clip
        self.quantiles_low_ = None
        self.quantiles_high_ = None
        self.scale_ = None
        self.offset_ = None

    def fit(self, X, y=None):
        """计算各维度的分位数和缩放参数"""
        X = np.asarray(X)
        self.quantiles_low_ = np.quantile(X, self.lower_quantile, axis=0)
        self.quantiles_high_ = np.quantile(X, self.upper_quantile, axis=0)

        # 计算缩放参数
        self.scale_ = (self.target_max - self.target_min) / (
            self.quantiles_high_ - self.quantiles_low_ + 1e-8
        )  # 避免除零
        self.offset_ = self.target_min - self.quantiles_low_ * self.scale_

        return {
            "quantiles_low_": self.quantiles_low_,
            "quantiles_high_": self.quantiles_high_,
            "scale_": self.scale_,
            "offset_": self.offset_,
        }

    def transform(self, X):
        """应用归一化"""
        X = np.asarray(X)
        X_norm = X * self.scale_ + self.offset_

        if self.clip:
            np.clip(X_norm, self.target_min, self.target_max, out=X_norm)

        return X_norm

    def inverse_transform(self, X_norm):
        """反归一化"""
        X_norm = np.asarray(X_norm)
        return (X_norm - self.offset_) / self.scale_


def _to_serialisable(obj):
    """Convert numpy types to JSON-serialisable Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    raise TypeError(f"Type {type(obj)} not serialisable")


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

    # 计算相对旋转 ΔR = R_{t+1} * R_t^{-1}
    delta_R = R_t_plus_1 @ R_t.transpose(0, 2, 1)

    # 转换为axis-angle
    delta_axis_angle = R.from_matrix(delta_R).as_rotvec()

    return delta_axis_angle


# 创建归一化器
normalizer = AdvancedQuantileNormalizer(
    lower_quantile=0.01, upper_quantile=0.99, target_range=(-1, 1)
)

error_list = []
action_list = []
state_list = []
all_tasks = []

# 要过滤的 task_id 列表（基于 task_id 过滤，而不是 job_id/episode_id）
# 例如: drop_list = {1901, 1902} 会过滤掉 task_id 为 1901 和 1902 的所有轨迹
drop_list = set([1765, 1790, 1830, 1832])

# 配置：根据任务选择不同的 job_id 范围（目录名范围）
# 注意：这里配置的是 job_id 范围，不是 task_id！
# pour (倒水): job_id 2218-2575
# open_close (开关抽屉): job_id 2846-3112
# erase (擦黑板): job_id 3149-3153
# pnp (抓取放置): job_id 1338-2188
# pour_coffee_105 (倒咖啡): job_id 3924-3928
TASK_NAME = "pour_coffee_105"  # 修改这里来切换任务
TASK_RANGES = {
    "pour": (2218, 2575),
    "open_close": (2846, 3112),
    "erase": (3149, 3153),
    "pnp": (1338, 2188),
    "pour_coffee_105": (3924, 3928),
}

START_TASK_ID, END_TASK_ID = TASK_RANGES[TASK_NAME]
DATA_DIR = "/share/project/caomingyu/a2d_data"

print(f"扫描数据目录: {DATA_DIR}")
print(f"任务类型: {TASK_NAME}")
print(f"Task ID 范围: {START_TASK_ID} - {END_TASK_ID}")

# 扫描数据目录，从标注文件加载任务数据
for job_id_str in os.listdir(DATA_DIR):
    try:
        job_id_num = int(job_id_str)
        # 检查 job_id 是否在配置的范围内
        if job_id_num >= START_TASK_ID and job_id_num <= END_TASK_ID:
            # 如果 job_id 在 drop_list 中，跳过
            if job_id_num in drop_list or job_id_str in drop_list:
                print(f"跳过 Job ID {job_id_str} (在过滤列表中)")
                continue

            # 加载标注文件 {job_id}.json
            annotation_file = os.path.join(DATA_DIR, job_id_str, f"{job_id_str}.json")
            if os.path.exists(annotation_file):
                try:
                    with open(annotation_file, "r", encoding="utf-8") as f:
                        task_list = json.load(f)  # 标注文件是数组格式
                        for task_data in task_list:
                            # 确保类型一致（统一为字符串，便于后续处理）
                            if "job_id" in task_data:
                                task_data["job_id"] = str(task_data["job_id"])
                            if "task_id" in task_data:
                                task_data["task_id"] = str(task_data["task_id"])
                            if "episode_id" in task_data:
                                task_data["episode_id"] = str(task_data["episode_id"])

                            all_tasks.append(task_data)
                except Exception as e:
                    print(f"加载标注文件失败 {annotation_file}: {e}")
            else:
                print(f"标注文件不存在: {annotation_file}，跳过 Job ID {job_id_str}")
    except (ValueError, OSError) as e:
        # 跳过非数字的目录名或其他错误
        continue

print(f"找到 {len(all_tasks)} 个任务")

for task in tqdm(all_tasks, desc="处理数据"):
    job_id = task["job_id"]
    task_id = task["task_id"]
    episode_id = task["episode_id"]
    sn_code = task.get("sn_code", "")

    hdf5_path = os.path.join(DATA_DIR, str(job_id), str(sn_code), str(episode_id))

    try:
        with h5py.File(os.path.join(hdf5_path, "aligned_joints.h5"), "r") as f:
            state_eepose_pos = f["action"]["end"]["position"][:]
            # 四元数转6D
            state_eepose_left_orientation = quat_to_6d(
                f["action"]["end"]["orientation"][:, 0, :]
            )
            state_eepose_right_orientation = quat_to_6d(
                f["action"]["end"]["orientation"][:, 1, :]
            )
            # 夹爪反转: 0关1开
            state_eepose_left_gripper = 1 - f["action"]["left_effector"]["position"][:]
            state_eepose_right_gripper = (
                1 - f["action"]["right_effector"]["position"][:]
            )

        # 构建state_eepose (右臂在前，左臂在后): 3+6+1 + 3+6+1 = 20维
        state_eepose = np.concatenate(
            [
                state_eepose_pos[:, 1, :],  # 右臂位置 (3)
                state_eepose_right_orientation,  # 右臂姿态6D (6)
                state_eepose_right_gripper,  # 右臂夹爪 (1)
                state_eepose_pos[:, 0, :],  # 左臂位置 (3)
                state_eepose_left_orientation,  # 左臂姿态6D (6)
                state_eepose_left_gripper,  # 左臂夹爪 (1)
            ],
            axis=-1,
        )

        # 计算action delta（使用整个轨迹，与旧版本 normal_a2d_0919.py 保持一致）
        # 注意：归一化参数计算使用整个轨迹，而不是标注范围
        action_eepose = np.concatenate(
            [
                state_eepose_pos[1:, 1, :]
                - state_eepose_pos[:-1, 1, :],  # 右臂位置delta (3)
                compute_d6_axis_angle_deltas(
                    state_eepose_right_orientation
                ),  # 右臂姿态delta (3)
                state_eepose_right_gripper[1:],  # 右臂夹爪 (1)
                state_eepose_pos[1:, 0, :]
                - state_eepose_pos[:-1, 0, :],  # 左臂位置delta (3)
                compute_d6_axis_angle_deltas(
                    state_eepose_left_orientation
                ),  # 左臂姿态delta (3)
                state_eepose_left_gripper[1:],  # 左臂夹爪 (1)
            ],
            axis=-1,
        )  # 总维度: 3+3+1 + 3+3+1 = 14维

        action_list.extend(action_eepose)
        state_list.extend(state_eepose[:-1])  # 去掉最后一帧

    except Exception as e:
        error_list.append(
            {
                "task_id": task_id,
                "job_id": job_id,
                "episode_id": episode_id,
                "error": str(e),
            }
        )
        continue

# 创建输出目录
norm_dir = "/share/project/fengli/code/norm"
os.makedirs(norm_dir, exist_ok=True)

# 输出文件路径
norm_file = os.path.join(norm_dir, f"a2d_norm_{TASK_NAME}.json")
error_file = os.path.join(norm_dir, f"error_{TASK_NAME}.json")

result_dict = {}
action_numpy = np.array(action_list)
state_numpy = np.array(state_list)
print(f"Action shape: {action_numpy.shape}")
print(f"State shape: {state_numpy.shape}")

action_normalizer = normalizer.fit(action_numpy)
state_normalizer = normalizer.fit(state_numpy)

result_dict["action.eepose"] = {
    k: _to_serialisable(v) for k, v in action_normalizer.items()
}
result_dict["state.eepose"] = {
    k: _to_serialisable(v) for k, v in state_normalizer.items()
}

with open(norm_file, "w", encoding="utf-8") as f:
    json.dump(result_dict, f, indent=4, ensure_ascii=False)

with open(error_file, "w", encoding="utf-8") as f:
    json.dump(error_list, f, indent=4, ensure_ascii=False)

print(f"\n✅ 归一化参数已保存: {norm_file}")
print(f"✅ 错误记录已保存: {error_file}")
