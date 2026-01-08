import dataclasses
import logging
import os
import io
import pandas as pd
import numpy as np
import base64
import queue
import threading

from websocket_client import WebsocketClientPolicy
from torchvision import transforms
from PIL import Image

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

to_tensor = transforms.ToTensor()

# 测试客户端参数
@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 5001
    data_file: str = "/home/admin123/Desktop/episode_000005.parquet"  # 替换为实际数据文件路径
    test_interval: int = 50

class TestClient:
    def __init__(self, args: Args):
        self.shared_queue = queue.Queue()
        self.args = args
        self.ws_client = WebsocketClientPolicy(message_queue=self.shared_queue, host=self.args.host, port=self.args.port)
        self.df = self._load_data(self.args.data_file)
        self.buffer = []
        self._stop_event = None
        self._thread = None
        self.true_action = None

    def _load_data(self, file_path: str):
        try:
            logger.info(f"加载数据文件: {file_path}")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"数据文件不存在: {file_path}")
            df = pd.read_parquet(file_path)
            logger.info(f"数据文件加载成功，共{len(df)}行数据")
            return df
        except Exception as e:
            logger.error(f"加载数据失败: {e}")
            raise

    def decode_image_base64(self, image_base64):
        """解码base64图片"""
        try:
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            image = to_tensor(image)
            return image
        except Exception as e:
            logger.error(f"图片解码失败: {e}")
            raise ValueError(f"图片解码失败: {e}")

    def process_images(self, images_dict):
        """处理图片列表，适配openpi的图片格式"""
        try:
            sample_dict = {}
            # 根据openpi的配置调整图片键名
            for k in ['cam_high', 'cam_left_wrist', 'cam_right_wrist']:
                if k in images_dict:
                    sample_dict[k] = self.decode_image_base64(images_dict[k])
                else:
                    logger.warning(f"缺少图片: {k}")

        except Exception as e:
            logger.error(f"处理图片失败: {e}")
            raise ValueError(f"处理图片失败: {e}")

        return sample_dict

    def _build_obs_from_row(self, row_data, row_index):
        """从数据行构建测试数据"""
        try:
            # 提取状态数据
            if 'observation.state' in row_data:
                qpos = row_data['observation.state']
                if isinstance(qpos, (list, np.ndarray)):
                    qpos = [qpos] if len(qpos.shape) == 1 else qpos.tolist()
            else:
                print("observation.state not found")
                return None
            
            images = {}
            
            cam_mapping = {
                'observation.images.cam_high': 'cam_high',
                'observation.images.cam_left_wrist': 'cam_left_wrist', 
                'observation.images.cam_right_wrist': 'cam_right_wrist'
            }
            
            for parquet_key, api_key in cam_mapping.items():
                if parquet_key in row_data and pd.notna(row_data[parquet_key]['bytes']):
                    # 处理图片数据 - 将bytes转换为base64字符串
                    if isinstance(row_data[parquet_key]['bytes'], bytes):
                        # 如果是bytes类型，进行base64编码
                        import base64
                        images[api_key] = base64.b64encode(row_data[parquet_key]['bytes']).decode('utf-8')
                    elif isinstance(row_data[parquet_key]['bytes'], str):
                        # 如果已经是字符串，直接使用
                        images[api_key] = row_data[parquet_key]['bytes']
                    else:
                        print(f"图片数据格式错误: {parquet_key}, 类型: {type(row_data[parquet_key]['bytes'])}")
                        return None

            
            # 构建最终的测试数据
            test_data = {
                "state": qpos,
                "eef_pose": [[0.0] * 7],  # 添加必需的eefpose字段，7维位姿
                "images": images,
                "prompt": "pick up the orange and put it into the basket"
            }

            # 提取真实的action数据用于对比
            if 'action' in row_data:
                action_data = row_data['action']
                # 检查action_data是否有效
                if action_data is not None and (isinstance(action_data, np.ndarray) and action_data.size > 0) or (not isinstance(action_data, np.ndarray) and action_data):
                    if isinstance(action_data, np.ndarray):
                        action_data = action_data.tolist()
                else:
                    logger.warning(f"第{row_index}行action数据为空")
            else:
                logger.warning(f"第{row_index}行缺少action字段")

            if action_data is not None:
                logger.info(f"提取真实action数据，维度: {len(action_data) if action_data else 0}")
            else:
                logger.warning("无法提取action序列，无法进行MSE对比")
            
            logger.info(f"构建的测试数据包含:")
            logger.info(f"  - qpos维度: {len(qpos)} x {len(qpos[0]) if qpos else 0}")
            logger.info(f"  - 图片数量: {len(images)}")
            
            return test_data
            
        except Exception as e:
            logger.error(f"构建测试数据失败: {e}")
            raise


    # async def run_inference(self):
    #     """运行推理并与服务端交互"""
    #     # 随机抽取一个 episode
    #     index = np.random.randint(0, len(self.df))
    #     logger.info(f"测试第 {index} 个 episode...")

    #     # 准备数据
    #     row_data = self.df.iloc[index]
    #     obs = self._build_obs_from_row(row_data)
    #     logger.info(f"State原始数据: {obs['state']}")
    #     logger.info(f"State维度: {obs['state'].shape}")

    #     # 发送推理请求
    #     try:
    #         logger.info("发送推理请求...")
    #         response = await self.ws_client.infer(obs)  # WebSocket 发送数据
    #         logger.info(f"推理结果: {response}")
    #     except Exception as e:
    #         logger.error(f"推理失败: {e}")

    # def run(self):
    #     """运行客户端"""
    #     loop = asyncio.get_event_loop()
    #     loop.run_until_complete(self.run_inference())



    def _start_drain_thread(self):
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._drain_actions)
        self._thread.start()

    def _drain_actions(self):
        while not self._stop_event.is_set():
            try:
                payload = self.shared_queue.get(timeout=0.1)

                if isinstance(payload, dict) and "actions" in payload:
                    arr = payload["actions"]  # numpy.ndarray shape (50, 14)
                    # 转成 Python list-of-lists
                    actions = arr.tolist()
                elif isinstance(payload, (list, tuple)):
                    actions = list(payload)
                else:
                    logger.error(f"Unsupported payload: {type(payload)}")
                    actions = []

                # 只取前 50 条
                self.buffer = actions[:50]

                logger.info(
                    f"换包：收到新动作 {len(self.buffer)} 条，每条长度 {len(self.buffer[0]) if self.buffer else 0}"
                )

            except queue.Empty:
                continue

    def start(self):
        row_data = self.df.iloc[0]
        obs = self._build_obs_from_row(row_data, 0)

        # logger.info(f"发送初始观测: {obs}")
        logger.info(f"发送初始观测")
        self.ws_client.send_obs(obs)
        if self._thread is None or not self._thread.is_alive():
            self._start_drain_thread()

        for i in range(1, len(self.df)):
            while not self.buffer:
                # logger.info("等待 buffer 填充数据...")
                pass

            action = self.buffer.pop(0)
            print(f"infer_actions: {action}")
            print(f"truth_actions: {self.df.iloc[i]['action']}")

            obs_i = self._build_obs_from_row(self.df.iloc[i], i)
            self.ws_client.send_obs(obs_i)

        logger.info("该 episode 已按顺序送完所有观测。")


if __name__ == "__main__":
    # 设置参数
    args = Args(data_file="/home/admin123/Desktop/episode_000005.parquet")  # 修改为实际数据路径
    client = TestClient(args)
    client.start()
