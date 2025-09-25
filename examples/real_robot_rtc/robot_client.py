import dataclasses
import numpy as np
import base64
import queue
import threading
import ipdb
import time
import cv2

from websocket_client import WebsocketClientPolicy
from robot_env import RobotEnv

# 测试客户端参数
@dataclasses.dataclass
class Args:
    host: str = "172.16.13.99"
    port: int = 8000
    test_interval: int = 50

# 控制模式
CONTROL_MODE = 'joint'  # eepose, joint

# 图像编码
def encode_image(img: np.ndarray) -> str:
    """Encode OpenCV image as base64 PNG string."""
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')

# 编码为 JPEG bytes
def encode_image_bytes(img: np.ndarray) -> bytes:
    h, w = img.shape[:2]
    img_small = cv2.resize(img, (w // 2, h // 2))
    ok, buf = cv2.imencode('.jpg', img_small, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise RuntimeError("cv2.imencode('.jpg', ...) failed")
    return bytes(buf)

# 从帧中编码图像
def encode_image_from_frame(frames, cam_name):
    img = frames.get(cam_name)

    # 在编码前检查图像是否为空
    if img is not None and img.size > 0:
        # 如果图像有效，则进行编码
        # encoded_image = encode_image(img)
        encoded_image = encode_image_bytes(img)
        return encoded_image
    else:
        # 如果图像无效，打印一条警告信息或进行其他处理
        print(f"警告：无法获取 {cam_name} 的图像，或者图像为空。")
        return None
    
# 初始化 RobotEnv
env = RobotEnv(
    orbbec_serials=[0, 1, 2],
    # realsense_serials=[0],
    arm_ip="can0+can1"
)

# 初始位置
joint_command = [ 0.25434  ,  1.8082911, -1.327784 ,  0.7385629,  0.9807441, -0.199704 ,   0.6517 ,
                  -0.20462333,  1.6163499 , -1.0250355 , -0.92851543,  0.7400108 ,  0.37674767, 0.693,]
time.sleep(2)
env.control(joint_command)
time.sleep(6)

class TestClient:
    def __init__(self, args: Args):
        self.shared_queue = queue.Queue()
        self.args = args
        self.ws_client = WebsocketClientPolicy(message_queue=self.shared_queue, host=self.args.host, port=self.args.port)
        self.buffer = []
        self._stop_event = None
        self._thread = None
        self.true_action = None
        # self.delay = 0
        self.count = 1
        self._first = False

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
                    print(f"time: ", payload["policy_timing"])
                elif isinstance(payload, (list, tuple)):
                    actions = list(payload)
                else:
                    print(f"Unsupported payload: {type(payload)}")
                    actions = []

                # delay = self.count - self.delay
                if self._first is False:
                    self.buffer = actions
                    self._first = True
                else:
                    self.buffer = actions[5:]
                print(f"-----------------换包：收到新动作 {len(self.buffer)} 条-----------------")
                # self.delay = self.count

            except queue.Empty:
                continue

    def _wait_obs(self) -> dict:
        while True:
            frames, state = env.update_obs_window()

            if state is None or not frames:
                print("[!] 无状态或图像数据，跳过本轮")
                time.sleep(1)
                continue

            # 保存最新图像到当前目录
            for name, img in frames.items():
                save_path = f"./{name}_latest.png"
                cv2.imwrite(save_path, img)
                # print(f"[Saved] {save_path}")

            qpos = state["qpos"]
            qpos = np.concatenate([qpos[7:], qpos[:7]])
            encoded_images = {
                "cam_high": encode_image_from_frame(frames, "orbbec_2"),
                "cam_left_wrist": encode_image_from_frame(frames, "orbbec_1"),
                "cam_right_wrist": encode_image_from_frame(frames, "orbbec_0"),
            }
            data = {
                "state": [qpos.tolist()],           # shape: [1, 14]
                "images": encoded_images
            }
            return data

    def rollout(self):
        try:
            data = self._wait_obs()
            print("发送初始观测")
            self.ws_client.send_obs(data)
            if self._thread is None or not self._thread.is_alive():
                self._start_drain_thread()

            while not self._stop_event.is_set():
                while not self.buffer:
                    # print("等待 buffer 填充数据...")
                    time.sleep(0.1)
                
                action = self.buffer.pop(0)
                action = np.concatenate([action[7:], action[:7]])
                # print(f"[→ Step {self.count}] 执行动作: {action.round(3)}")
                print(f"[→ Step {self.count}] 执行动作---")
                t0 = time.time(); env.control(action); dt = time.time()-t0
                print(f"执行耗时 {dt*1000:.1f} ms")
                t0 = time.time(); last_obs = self._wait_obs(); dt = time.time()-t0
                print(f"观测耗时 {dt*1000:.1f} ms")
                self.ws_client.send_obs(data)
                self.count += 1
                # time.sleep(0.05)

            # BATCH_SIZE = 20
            # while not self._stop_event.is_set():
            #     while not self.buffer:
            #         # print("等待 buffer 填充数据...")
            #         time.sleep(0.1)

            #     # 连续执行 n 个动作
            #     for i in range(BATCH_SIZE):
            #         action = self.buffer.pop(0)
            #         action = np.concatenate([action[7:], action[:7]])
            #         print(f"[→ Step {self.count}] 执行动作: {action.round(3)}")
            #         t0 = time.time(); env.control(action); dt = time.time()-t0
            #         print(f"执行耗时 {dt*1000:.1f} ms")
            #         self.count += 1

            #     # 只在最后一次执行后发送信息
            #     t0 = time.time(); last_obs = self._wait_obs(); dt = time.time()-t0
            #     print(f"观测耗时 {dt*1000:.1f} ms")
            #     self.ws_client.send_obs(last_obs)
            #     print(f"[✓] 本批({BATCH_SIZE})步执行完毕，已发送最后一次观测。")
            #     self.buffer.clear()

        finally:
            self._stop_event.set()
            if self._thread.is_alive():
                self._thread.join(timeout=1.0)
            env.shutdown()
            print("[Main] RobotEnv shut down.")

if __name__ == "__main__":
    args = Args()
    client = TestClient(args)
    client.rollout()
