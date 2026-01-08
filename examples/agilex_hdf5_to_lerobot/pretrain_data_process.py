import os
import logging
import json
import numpy as np
import cv2
import h5py
from tqdm import tqdm
from PIL import Image
import jsonlines
from typing import Tuple, List, Dict, Any, Union
import sys
sys.path.append("/share/project/dumengfei/code/sim_data_process")
sys.path.append("/share/project/dumengfei/code/pretrain_data_process")
from pose_transform import quat_to_6d, compute_d6_axis_angle_deltas
from action_token.action_chunk_to_fast_token import ActionChunkProcessor

FRAME_SAMPLE_INTERVAL = os.getenv("FRAME_SAMPLE_INTERVAL", 3)
ACTION_SAMPLE_INTERVAL = os.getenv("ACTION_SAMPLE_INTERVAL", 1)
PADDING = os.getenv("PADDING", 0)
DATA_VERSION = os.getenv("DATA_VERSION", 'data_test')
FRAME_SAMPLE_INTERVAL = int(FRAME_SAMPLE_INTERVAL)
ACTION_SAMPLE_INTERVAL = int(ACTION_SAMPLE_INTERVAL)
PADDING = int(PADDING)

_TOKENIZER_CACHE: dict[int, ActionChunkProcessor] = {}
def get_tokenizer(max_len: int) -> ActionChunkProcessor:
    """Return a cached ActionChunkProcessor (one per process).

    每个 Ray worker 进程各自维护 _TOKENIZER_CACHE，首次调用时才实例化。
    """
    tok = _TOKENIZER_CACHE.get(max_len)
    if tok is None:
        tok = ActionChunkProcessor(max_len=max_len)
        _TOKENIZER_CACHE[max_len] = tok
        logger.debug("Tokenizer initialised in PID %s (max_len=%s)", os.getpid(), max_len)
    return tok

# ----------------------------------------------------------------------------
# logging config
# ----------------------------------------------------------------------------

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

def get_libero_dummy_action():
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return np.array([0, 0, 0, 0, 0, 0, -1])

def find_last_duplicate_start(lst):
    if not lst:  # 处理空列表情况
        return -1
    
    last_element = lst[-1]
    # 从倒数第二个元素开始向前遍历
    for i in range(len(lst)-2, -1, -1):
        if lst[i] != last_element:
            # 找到第一个不同的元素，返回下一个索引
            if i + 1 == len(lst) - 1:
                return 0
            else:
                return i + 1
    return 0

class TrajectoryProcessor:
    def __init__(self, args):
        """初始化轨迹处理器
        
        Args:
            args: 包含各种配置参数的对象
        """
        self.args = args
        self._load_normalization_parameters()
        # self.action_tokenizer = get_tokenizer(self.args.max_len)  # 假设get_tokenizer已定义
        # 可以在这里初始化其他需要的属性

    def _load_normalization_parameters(self):
        """加载归一化参数"""
        with open(self.args.normal_path, 'r', encoding='utf-8') as f:
            norm_para = json.load(f)
        
        self.action_eepose_scale = np.array(norm_para["action.eepose"]["scale_"])
        self.action_eepose_offset = np.array(norm_para["action.eepose"]["offset_"])
        self.action_qpos_scale = np.array(norm_para["action.qpos"]["scale_"])
        self.action_qpos_offset = np.array(norm_para["action.qpos"]["offset_"])

    def transform(self, x: np.array, scale: np.array, offset: np.array, clip: bool = True) -> np.array:
        """数据转换（归一化）
        
        Args:
            x: 原始数据
            scale: 缩放因子
            offset: 偏移量
            clip: 是否裁剪到[-1, 1]范围
            
        Returns:
            转换后的数据
        """
        x_norm = x * scale + offset
        if clip:
            np.clip(x_norm, -1, 1, out=x_norm)  
        return x_norm

    def inverse_transform(self, x_norm: np.array, scale: np.array, offset: np.array) -> np.array:
        """逆转换（从归一化数据恢复原始数据）
        
        Args:
            x_norm: 归一化后的数据
            scale: 缩放因子
            offset: 偏移量
            
        Returns:
            原始数据
        """
        x_norm = np.asarray(x_norm)
        return (x_norm - offset) / scale

    def save_frames(
        self,
        agentview_rgb_matrix: np.array,
        eye_in_hand_rgb_matrix: np.array,
        output_dir: str,
        image_format: str = "jpg",
    ) -> Tuple[int, List[str]]:
        """保存多个视频的帧并返回写入的帧数
        
        Returns:
            Tuple[int, List[str]]: (写入的帧数, 路径列表)
        """
        N = agentview_rgb_matrix.shape[0]
        image_paths = []
        
        for i in range(N):
            # 生成文件名 (按帧编号命名)
            agentview_filename = os.path.join(output_dir, f"agentview_{i}.{image_format}")
            eye_in_hand_filename = os.path.join(output_dir, f"eye_in_hand_{i}.{image_format}")
            
            try:
                # 保存主视角帧 (OpenCV 需要 BGR 格式)
                agentview_frame = agentview_rgb_matrix[i][::-1, ::-1]
                agentview_frame_bgr = cv2.cvtColor(agentview_frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(agentview_filename, agentview_frame_bgr)
                
                # 保存手眼视角帧
                eye_in_hand_frame = eye_in_hand_rgb_matrix[i][::-1, ::-1]
                eye_in_hand_frame_bgr = cv2.cvtColor(eye_in_hand_frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(eye_in_hand_filename, eye_in_hand_frame_bgr)
                
                image_paths.extend([agentview_filename, eye_in_hand_filename])
                
            except Exception as e:
                print(f"保存第 {i} 帧失败: {str(e)}")
                continue

        return len(image_paths), image_paths

    def json_fill(self, 
                raw_task: str, 
                task: str,
                image_path_list: List[str], 
                action_eepose_tokenizer_path, 
                state_eepose_path, 
                action_eepose_path,
                action_qpos_tokenizer_path,
                state_qpos_path,
                action_qpos_path
            ):
        """生成JSON条目
        
        Args:
            lan: 自然语言描述
            image_path_list: 图像路径列表
            action_token_path: 动作令牌路径
            
        Returns:
            生成的JSON字典
        """
        action_str_list = ['<action_token>'] * 1
        action_str = '<action_split>'.join(action_str_list)
        json_item = {
                "raw_task": raw_task,
                "task": task,
                "image": image_path_list,
                "action_eepose_token": action_eepose_tokenizer_path,
                "action_qpos_token": action_qpos_tokenizer_path,
                "state":{
                    "eepose": state_eepose_path,
                    "qpos": state_qpos_path,
                },
                "action":{
                    "eepose": action_eepose_path,
                    "qpos": action_qpos_path,
                },
                "conversations": [
                    {
                        "from": "human",
                        "value": f"According to the robot front image<image> and robot wrist image<image>, what action should the robot take to complete: {task}."
                    },
                    {
                        "from": "gpt",
                        "value": action_str
                    }
                ]
            }
        return json_item

    def extract_num_frames(self, video_path):
        # Extracts video frames at 1 fps
        cap = cv2.VideoCapture(video_path)
        vid_fps = cap.get(cv2.CAP_PROP_FPS)
        round_vid_fps = round(vid_fps)
        num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        return int(num_frames),vid_fps, cap

    def process_single_item(self, sub_dir_path, sub_dir_path_image):
        """Process a single trajectory.

        The function is deliberately defensive: *every* IO operation is protected so that an
        isolated failure does not crash the entire job.
        """
        
        action_tokenizer = get_tokenizer(self.args.max_len)
        for data_item in tqdm(self.all_tasks[self.args.task_start_idx: self.args.task_end_idx]):
            try:
                self.total_traj += 1
                # ---------------------------------------------------------------------
                # directory setup
                # ---------------------------------------------------------------------
                job_id = data_item['job_id']
                task_id = data_item['task_id']
                episode_id = data_item['episode_id']
                hdf5_path = data_item['aligned_data_path'].replace(f'zhiyuan-frame://genie-data-base-zhiyuan-1347244451/framing/GENIE/{task_id}/{job_id}', f'/share/project/caomingyu/a2d_data/{job_id}')
                cam_path = os.path.join(hdf5_path, 'camera')
                with h5py.File(os.path.join(hdf5_path, 'aligned_joints.h5'), "r") as f:
                    state_eepose_pos = f['action']['end']['position'][:]
                    state_eepose_left_orientation = quat_to_6d(f['action']['end']['orientation'][:, 0, :])
                    state_eepose_right_orientation = quat_to_6d(f['action']['end']['orientation'][:, 1, :])
                    state_eepose_left_gripper = 1-f['action']['left_effector']['position'][:]
                    state_eepose_right_gripper = 1-f['action']['right_effector']['position'][:]
                    # 改为0关1开
                    state_qpos = f["state"]["joint"]["position"][:]
                    state_qpos_left_gripper = f['state']['left_effector']['position'][:]
                    state_qpos_right_gripper = f['state']['right_effector']['position'][:]
                    # get action
                    action = np.concatenate(
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
                    qpos = np.concatenate((state_qpos[:,7:], state_qpos_right_gripper, state_qpos[:,:7], state_qpos_left_gripper), axis=1)

                uuid = f"{job_id}_{task_id}_{episode_id}"
                images_path = os.path.join(sub_dir_path_image, "images", uuid)
                action_token_path = os.path.join(sub_dir_path, "action_token", uuid)
                os.makedirs(images_path, exist_ok=True)
                os.makedirs(action_token_path, exist_ok=True)

                raw_task = data_item["task_name"]
                sub_task = data_item["task_name"]
                # sub_task_configs = data_item['label_info']['action_config']
                json_entries: List[dict] = []

                # for sub_task_config in sub_task_configs:
                # sub_task = sub_task_config['action_text']
                start = data_item['label_info']['action_config'][0]['start_frame']
                end = data_item['label_info']['action_config'][-1]['end_frame']

                # ------------------------------------------------------------------
                # extract and write frames
                # ------------------------------------------------------------------
                # 过滤相邻不变的动作并记录原始索引
                filtered_action = []
                filtered_qpos = []
                original_indices = []  # 记录过滤后动作对应的原始索引

                # 保留第一帧
                # 基于过滤后的数据重新计算循环范围
                loop_start = max(0, start)
                loop_end = max(loop_start, end+1)  # ensure non-negative

                self.original_sample_num += loop_end - loop_start
                filtered_action.append(action[loop_start])
                filtered_qpos.append(qpos[loop_start])
                original_indices.append(loop_start)
                last_action = action[loop_start]
                for i in range(loop_start+1, loop_end):
                    # 检查动作是否有变化
                    action_changed = not (np.allclose(action[i-1], action[i], rtol=1e-5, atol=1e-6))
                    # action_changed = not (np.all(np.abs(action[i] - last_action) < 1e-5))
                    
                    if action_changed:
                        filtered_action.append(action[i])
                        filtered_qpos.append(qpos[i])
                        original_indices.append(i)  # 记录变化帧的原始索引
                        last_action = action[i].copy()

                # 转换为numpy数组
                filtered_action = np.array(filtered_action)
                filtered_qpos = np.array(filtered_qpos)
                original_indices = np.array(original_indices)

                self.wo_static_sample_num += len(filtered_action)
                # 更新结束索引为过滤后的长度
                end = len(filtered_action)
                if end == 0:
                    logger.warning("No valid action data after filtering for item %s", uuid)
                    return []

                # ------------------------------------------------------------------
                # Build JSON entries
                # ------------------------------------------------------------------
                if len(filtered_action) < 15:
                    construct_num = 0
                    self.filtered_traj += 1
                else:
                    if self.args.padding is None:
                        construct_num = max(1, end-self.args.chunk)
                    else:
                        construct_num = max(1, end-self.args.padding)

                for i in range(construct_num):
                    # 选择分块索引 - 用min确保不超出边界
                    index = [min(i + j, end - 1) for j in range(self.args.chunk + 1)]
                    # 获取过滤后的数据块
                    action_chunk = filtered_action[index]
                    qpos_chunk = filtered_qpos[index]
                    # 获取对应的原始索引，用于匹配图像
                    original_index_chunk = original_indices[index]

                    # 计算delta – 确保形状匹配
                    try:
                        action_delta = np.concatenate(
                            [
                                action_chunk[1:, :3] - action_chunk[:-1, :3],
                                compute_d6_axis_angle_deltas(action_chunk[:, 3:9]),
                                action_chunk[1:, [9]],
                                action_chunk[1:, 10:13] - action_chunk[:-1, 10:13],
                                compute_d6_axis_angle_deltas(action_chunk[:, 13:19]),
                                action_chunk[1:, [19]],
                            ],
                            axis=-1,
                        )
                        qpos_delta = np.concatenate(
                            [
                                qpos_chunk[1:, :7] - qpos_chunk[:-1, :7],
                                qpos_chunk[1:, [7]],
                                qpos_chunk[1:, 8:15] - qpos_chunk[:-1, 8:15],
                                qpos_chunk[1:, [15]],
                            ],
                            axis=-1,
                        )
                    except Exception as exc:
                        logger.error("Failed to build action delta for item %s/%s – %s", uuid, i, exc)
                        continue
                        
                    nor_action_delta = self.transform(action_delta, self.action_eepose_scale, self.action_eepose_offset)
                    nor_action_qpos = self.transform(qpos_delta, self.action_qpos_scale, self.action_qpos_offset)

                    action_chunk = action_chunk[:-1]
                    qpos_chunk = qpos_chunk[:-1]
                    # 保存数组
                    # 保持0为关闭，大于1为打开
                    base_idx = original_indices[i]
                    action_eepose_tokenizer_path = os.path.join(action_token_path, f"action_eepose_token_{base_idx}.npy")
                    state_eepose_path = os.path.join(action_token_path, f"state_eepose_{base_idx}.npy")
                    action_eepose_path = os.path.join(action_token_path, f"action_eepose_{base_idx}.npy")

                    state_qpos_path = os.path.join(action_token_path, f"state_qpos_{base_idx}.npy")
                    action_qpos_path = os.path.join(action_token_path, f"action_qpos_{base_idx}.npy")
                    action_qpos_tokenizer_path = os.path.join(action_token_path, f"action_qpos_token_{base_idx}.npy")
                    # 保存原始索引，用于后续验证或处理
                    original_indices_path = os.path.join(action_token_path, f"original_indices_{base_idx}.npy")
                    # 使用inverse的gripper动作作为action token
                    action_eepose_token = action_tokenizer.process_action_chunk_to_fast_token(nor_action_delta)
                    action_qpos_token = action_tokenizer.process_action_chunk_to_fast_token(nor_action_qpos)

                    # eepose
                    np.save(action_eepose_tokenizer_path, action_eepose_token)
                    np.save(state_eepose_path, action_chunk)
                    np.save(action_eepose_path, action_delta)
                    # joint
                    np.save(action_qpos_tokenizer_path, action_qpos_token)
                    np.save(state_qpos_path, qpos_chunk)
                    np.save(action_qpos_path, qpos_delta)
                    # 保存原始索引
                    np.save(original_indices_path, original_index_chunk)  

                    image_path_list = []
                    for view in ["head_color", "hand_right_color", "hand_left_color"]:
                        ori_image_path = os.path.join(cam_path, str(base_idx),f"{view}.jpg")
                        img = Image.open(ori_image_path).convert('RGB')

                        os.makedirs(os.path.join(images_path, str(base_idx)), exist_ok=True)
                        img_resize = img.resize((320,240))
                        target_path = os.path.join(images_path, str(base_idx), f'{view}.jpg')
                        img_resize.save(target_path)
                        image_path_list.append(target_path)
                    
                    json_item = self.json_fill(
                        raw_task, 
                        sub_task, 
                        image_path_list, 
                        action_eepose_tokenizer_path, 
                        state_eepose_path, 
                        action_eepose_path,
                        action_qpos_tokenizer_path,
                        state_qpos_path,
                        action_qpos_path,
                    )
                    json_entries.append(json_item)


                try:
                    if json_entries[-1] in json_entries[::FRAME_SAMPLE_INTERVAL]:
                        sampled_entries = json_entries[::FRAME_SAMPLE_INTERVAL]
                    else:
                        sampled_entries = json_entries[::FRAME_SAMPLE_INTERVAL]
                        sampled_entries.append(json_entries[-1])

                    for json_item in sampled_entries:
                        json_line = json.dumps(json_item)
                        self.json_file_writer.write(json_line+'\n')
                except:
                    print(f"过滤静止动作{len(filtered_action)} < 15，跳过！")

            except Exception as exc:
                # 任何未处理的异常 - 报告并跳过此项目
                print(f"处理项目时发生异常 - {exc}")
                return []

    def process_task(self):
        self.original_sample_num = 0
        self.wo_static_sample_num = 0
        self.filtered_traj = 0
        self.total_traj = 0

        write_file_path = f"/share/project/dumengfei/data/{DATA_VERSION}/a2d_train_{self.args.task_start_idx}_{self.args.task_end_idx}.jsonl"
        # if os.path.exists(write_file_path):
        #     with jsonlines.open(write_file_path, "r") as f:
        #         data_num = sum([1 for _ in f])
            
        #     if data_num >= self.args.task_end_idx - self.args.task_start_idx:
        #         print("Finshed")
        #         return

        sub_dir_path = os.path.join(self.args.root_path, f"a2d_train_{self.args.task_start_idx}_{self.args.task_end_idx}")
        sub_dir_path_image = os.path.join(self.args.root_path, f"a2d_train_{self.args.task_start_idx}_{self.args.task_end_idx}")
        os.makedirs(sub_dir_path, exist_ok=True)
        self.json_file_writer = open(write_file_path, 'w')
        self.process_single_item(sub_dir_path, sub_dir_path_image)
        self.json_file_writer.close()
        print("Finshed")

        print(f"original_sample_num: {self.original_sample_num}\nwo_static_sample: {self.wo_static_sample_num}")
        print(f"total_traj: {self.total_traj}\nfiltered_traj: {self.filtered_traj}\n")

    def run(self):
        """运行主处理流程"""
        drop_list = [
            "683/22628", "680/22594", "680/22596", "679/22580", "679/22585", "679/22587",
            "678/22570", "678/22573", "678/22575", "677/22565", "676/22466", "676/22478",
            "676/22490", "675/22514", "675/22518", "675/22520", "675/22535", "675/22548",
            "675/22537", "668/22443", "685/22669", "684/22648", "684/22649", "683/22614",
            "683/22616", "683/22623", "767/29963", "767/29979", "767/30027", "767/30040",
            "762/29875", "762/29884", "762/29887", "762/29888", "762/29892", "762/29899",
            "762/29907", "762/29922", "848/32874", "848/32887", "848/32899", "848/32900",
            "848/32906", "848/32907", "848/32909", "848/32918", "848/32922", "848/32932",
            "775/30256", "775/30259", "775/30260", "775/30266", "775/30267", "775/30269",
            "775/30272", "775/30284", "778/30299", "778/30303", "778/30305", "778/30307",
            "778/30312", "778/30313", "778/30317", "778/30332", "807/30404", "848/32874",
            "848/32887", "848/32899", "848/32900", "848/32906", "848/32907", "848/32909",
            "848/32918", "848/32922", "848/32932", "854/32984", "854/32989", "854/32992",
            "854/33034", "854/33039", "854/33049", "848/32946",
            "885/33205", "885/33220", "885/33222", "885/33297", "949/34435", "945/34435",
            "964/34467", "956/34455", "970/34480", "982/34503", "982/34507", "970/34480",
            "982/34503", "982/34507", "1053/34764", "1051/34760", "1049/34743", "1048/34735"
        ]
        excluded_ids = [756, 762, 767, 769, 848, 854]
        task_json_paths = []
        for task_id in os.listdir(f"/share/project/caomingyu/a2d_data"):
            try:
                if int(task_id) >= 677 and int(task_id) <= 1053 and int(task_id) not in excluded_ids:
                    task_json_paths += [os.path.join("/share/project/caomingyu/a2d_data", task_id, _) for _ in os.listdir(f"/share/project/caomingyu/a2d_data/{task_id}/") if _.endswith('.json')]
            except:
                continue
        self.all_tasks = []
        all_tasks = []
        for task_json_path in task_json_paths:
            with open(task_json_path, 'r', encoding='utf-8') as file:
                all_tasks += json.load(file)
        os.makedirs(f"/share/project/dumengfei/data/{DATA_VERSION}", exist_ok=True)
        for task in all_tasks:
            job_id = task['job_id']
            task_id = task['task_id']
            episode_id = task['episode_id']
            if f"{job_id}/{episode_id}" in drop_list:
                continue
            else:
                self.all_tasks.append(task)
        self.process_task()
        
       
# 使用示例
if __name__ == "__main__":
    # 这里通常会解析命令行参数
    import argparse
    
    parser = argparse.ArgumentParser(description="轨迹数据处理器参数")
    # 添加必要的命令行参数
    parser.add_argument("--root_path", type=str, default=f"/share/project/dumengfei/data/{DATA_VERSION}/a2d_train_data_twin")
    parser.add_argument("--image_path", type=str, default=f"/share/project/dumengfei/data/{DATA_VERSION}/a2d_train_data_twin")
    parser.add_argument("--normal_path", type=str, default="/share/project/dumengfei/code/pretrain_data_process/real_data/a2d/a2d_normal_0920_30Hz.json")
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--padding", type=int, default=None)
    parser.add_argument("--chunk", type=int, default=30)
    parser.add_argument("--action", type=int, default=1)
    parser.add_argument("--task_start_idx", type=int, default=590)
    parser.add_argument("--task_end_idx", type=int, default=621)
    
    args = parser.parse_args()
    
    # 初始化并运行处理器
    processor = TrajectoryProcessor(args)
    processor.run()

