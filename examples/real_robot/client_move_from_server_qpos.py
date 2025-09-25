"""
if ! ifconfig | grep -q "can_left"; then
    source /home/agilex/cobot_magic/Piper_ros_private-ros-noetic/can_config.sh 
    sleep 5
fi
export PYTHONPATH=$PYTHONPATH:/home/agilex/1ms.ai/pyorbbecsdk/install/lib/:/home/agilex/1ms.ai/ugv_sdk
source /home/agilex/miniconda3/etc/profile.d/conda.sh  
pkill dora
conda activate dora
ulimit -n 1000000
"""
import base64
import io
import time
import numpy as np
import requests
import cv2
# import ipdb
from piper_sdk import C_PiperInterface
from robot_env import RobotEnv
import datetime
from PIL import Image

CONTROL_MODE = 'joint'  # eepose, joint
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
        # image_data = base64.b64decode(encoded_image)
        # img_array = np.fromstring(image_data,np.uint8)
        # image_data2 = cv2.imdecode(img_array,cv2.COLOR_BGR2RGB)
        # image = Image.fromarray(image_data2)
        # image.save("test_yunfan3.png")

        # image_pil = Image.fromarray(img)
        # image_pil.save("test_yunfan.png")
        return encoded_image
    else:
        # 如果图像无效，打印一条警告信息或进行其他处理
        print(f"警告：无法获取 {cam_name} 的图像，或者图像为空。")
        return None

# 初始化 RobotEnv（替换成你实际的相机 index 和机械臂 IP）
env = RobotEnv(
    # orbbec_serials=[0, 1, 2],
    realsense_serials=[0,1,2],
    arm_ip="can0+can1"
)

# 采集初始位置
joint_command = [ 0.25434  ,  1.8082911, -1.327784 ,  0.7385629,  0.9807441, -0.199704 ,   0.6517 ,
                  -0.20462333,  1.6163499 , -1.0250355 , -0.92851543,  0.7400108 ,  0.37674767, 0.693,]

# time.sleep(2)
# env.control(joint_command)
# print(f"action: {joint_command}")
time.sleep(6)

try:
    while True:
        time.sleep(2)
        frames, state = env.update_obs_window()
        
        if state is None or not frames:
            print("[!] 无状态或图像数据，跳过本轮")
            time.sleep(1)
            continue

        qpos = state["qpos"]  # 右+左
        eef_pose = state["eef_pose"]   # 右+左

        # qpos = np.concatenate([qpos[7:], qpos[:7]])
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        # 保存最新图像到当前目录
        for name, img in frames.items():
            save_path = f"./Log/{name}_{timestamp}.png"
            cv2.imwrite(save_path, img)
            save_path = f"./{name}_latest.png"
            cv2.imwrite(save_path, img)
            print(f"[Saved] {save_path}")

        encoded_images = {
            "cam_high": encode_image_from_frame(frames, "realsense_2"),
            "cam_left_wrist": encode_image_from_frame(frames, "realsense_0"),
            "cam_right_wrist": encode_image_from_frame(frames, "realsense_1"),
            # "cam_high_realsense": encode_image_from_frame(frames, "realsense_0"),
        }
        # import ipdb; ipdb.set_trace()
        data = {
            "qpos": [qpos.tolist()],           # shape: [1, 14]
            "eef_pose": [eef_pose.tolist()],  # shape: [1, 14]
            "instruction": "Pick up the orange",
            "images": encoded_images
        }

        # response = requests.get("http://172.16.20.203:5001/replay", json=data, timeout=60)
        # response = requests.post("http://172.16.17.185:8000/infer", json=data, timeout=60)


        response = requests.post("http://172.16.13.99:5003/infer", json=data, timeout=60)

        # response = requests.post("http://172.16.20.231:5001/infer", json=data, timeout=60)
        print("[√] Response:", response.status_code)

        # print("[RAW]:", response.text[:300]) 

        result = response.json()
        # print("[Response JSON]:", result)
        # print("[JSON keys]:", list(result.keys()), "qpos_len=", (len(result.get("qpos", [])) if isinstance(result.get("qpos"), list) else None))
        
        if CONTROL_MODE == 'eepose':
            actions = result.get("eepose", [])
            if not actions:
                print("[!] 未返回动作，跳过控制")
                continue
            actions = np.array(actions)[:10]   # 执行步数修改

            # 获取完整动作序列并依次执行
            for i, act in enumerate(actions):
                # ipdb.set_trace()
                action = np.array(act, dtype=np.float32)
                # # # 调换前7维（右手）和后7维（左手）
                # if action.shape[0] == 14:
                #     action = np.concatenate([action[7:14], action[0:7]])
                print(f"[→ Step {i+1}] 执行动作: {action.round(3)}")
                env.control_eef(action)

        elif CONTROL_MODE == 'joint':
            actions = result.get("qpos", [])
            # print("qpos: ",result.get("qpos", []))
            # print("actions: ", actions)
            
            if not actions:
                print("[!] 未返回动作，跳过控制")
                continue
            actions = np.array(actions)[:20]  #步数修改

            # 获取完整动作序列并依次执行
            for i, act in enumerate(actions):
                action = np.array(act, dtype=np.float32)
                # ipdb.set_trace()
            
                # # 调换前7维（右手）和后7维（左手）
                # if action.shape[0] == 14:
                #     action = np.concatenate([action[7:], action[:7]])
                action[6] = 10*action[6]
                action[13] = 10*action[13]
                # action[:] = np.concatenate([action[:,7:14], action[:, 0:7]], axis=1)
                print(f"[→ Step {i+1}] 执行动作: {action.round(3)}")
                env.control(action)
except KeyboardInterrupt:
    print("\n[Main] Interrupted by user.")
finally:
    env.shutdown()
    print("[Main] RobotEnv shut down.")