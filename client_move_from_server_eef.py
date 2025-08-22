import base64
import io
import time
import numpy as np
import requests
import cv2
import ipdb
from piper_sdk import C_PiperInterface
from robot_env import RobotEnv
import datetime
import threading
from collections import deque

CONTROL_MODE = 'joint'  # eepose
def encode_image(img: np.ndarray) -> str:
    """Encode OpenCV image as base64 PNG string."""
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')

def encode_image_from_frame(frames, cam_name):
    # 获取图像
    img = frames.get(cam_name)

    # 在编码前检查图像是否为空
    if img is not None and img.size > 0:
        # 如果图像有效，则进行编码
        encoded_image = encode_image(img)
        return encoded_image
    else:
        # 如果图像无效，打印一条警告信息或进行其他处理
        print(f"警告：无法获取 {cam_name} 的图像，或者图像为空。")
        return None

# 初始化 RobotEnv（替换成你实际的相机 index 和机械臂 IP）
env = RobotEnv(
    orbbec_serials=[0, 1, 2],
    # realsense_serials=[0],
    arm_ip="can0+can1"
)

joint_command = [ 0.25434  ,  1.8082911, -1.327784 ,  0.7385629,  0.9807441, -0.199704 ,   0.6517 ,
                  -0.20462333,  1.6163499 , -1.0250355 , -0.92851543,  0.7400108 ,  0.37674767, 0.693,
                    ]

print(f"action: {joint_command}")

# time.sleep(2)
# env.control(joint_command)
# time.sleep(1)

try:
    while True:
        # cmd = input("\n按下 'c' 继续一次推理和控制，或按 Ctrl+C 退出：")
        # if cmd.strip().lower() != 'c':
        #     print("[!] 非法输入，输入 'c' 开始下一步。")
        #     continue

        frames, state = env.update_obs_window()
        if state is None or not frames:
            print("[!] 无状态或图像数据，跳过本轮")
            time.sleep(1)
            continue

        qpos = state["qpos"]
        eef_pose = state["eef_pose"]

        now = datetime.datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        # 保存最新图像到当前目录
        for name, img in frames.items():
            save_path = f"./Log/{name}_{timestamp}.png"
            cv2.imwrite(save_path, img)
            print(f"[Saved] {save_path}")

        encoded_images = {
            "cam_high": encode_image_from_frame(frames, "orbbec_0"),
            "cam_left_wrist": encode_image_from_frame(frames, "orbbec_1"),
            "cam_right_wrist": encode_image_from_frame(frames, "orbbec_2"),
            # "cam_high_realsense": encode_image_from_frame(frames, "realsense_0"),
        }

        data = {
            "state": [qpos.tolist()],           # shape: [1, 14]
            "eef_pose": [eef_pose.tolist()],  # shape: [1, 14]
            "instruction": "pick up the orange",
            "images": encoded_images
        }



        response = requests.get("http://172.16.20.203:5001/replay", json=data, timeout=120)

        # response = requests.post("http://172.16.17.185:8000/infer", json=data, timeout=60)
        print("[√] Response:", response.status_code)

        result = response.json()
        print("[Response JSON]:", result)
        # ipdb.set_trace()
        
        if CONTROL_MODE == 'eepose':
            actions = result.get("eepose", [])
            if not actions:
                print("[!] 未返回动作，跳过控制")
                continue
            actions = np.array(actions)[:20]

            # 获取完整动作序列并依次执行
            for i, act in enumerate(actions):
                # ipdb.set_trace()
                action = np.array(act, dtype=np.float32)
                # # 调换前7维（右手）和后7维（左手）
                # if action.shape[0] == 14:
                #     action = np.concatenate([action[7:14], action[0:7]])
                print(f"[→ Step {i+1}] 执行动作: {action.round(3)}")
                env.control_eef(action)
                # env.control(action)
                time.sleep(0.1)  # 可根据实际需要调整间隔时间
        elif CONTROL_MODE == 'joint':
            actions = result.get("qpos", [])
            
            if not actions:
                print("[!] 未返回动作，跳过控制")
                continue
            actions = np.array(actions)[:]

            # 获取完整动作序列并依次执行
            for i, act in enumerate(actions):
                action = np.array(act, dtype=np.float32)
                ipdb.set_trace()
            
                # # 调换前7维（右手）和后7维（左手）
                # if action.shape[1] == 14:
                # for single_action in action:

                action[:] = np.concatenate([action[:,7:14], action[:, 0:7]], axis=1)
                print(f"[→ Step {i+1}] 执行动作: {action.round(3)}")
                env.control(action)
                time.sleep(0.1)  # 可根据实际需要调整间隔时间
except KeyboardInterrupt:
    print("\n[Main] Interrupted by user.")
finally:
    env.shutdown()
    print("[Main] RobotEnv shut down.")
