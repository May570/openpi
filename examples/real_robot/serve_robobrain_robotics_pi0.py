"""
POST /infer 输入样例：
{
  "qpos": [[0.1, 0.2, ..., 0.3]],  # shape: [B, state_dim]，可为一维或二维数组
  "eef_pose": [[0.1, 0.2, ..., 0.3]],  # shape: [B, action_dim]，可为一维或二维数组
  "instruction": "请让机器人前进并避开障碍物",
  "images": [
    {
      "base_0_rgb": "<base64字符串>",
      "left_wrist_0_rgb": "<base64字符串>"
    }
    # 可以有多个样本，每个样本是一个相机名到base64图片的字典
  ],
}
"""

"""huaihai path:
/share/project/lvhuaihai# conda activate envs/openpi/
"""
"""
201行 offline 和 real_robot 有不同
"""
import os
import io
import base64
from PIL import Image
import sys
import torch
import h5py
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import time
import traceback
from torchvision import transforms
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
import threading

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# 启用CORS，允许跨域请求
CORS(app)

# 服务配置
SERVICE_CONFIG = {
    'host': '0.0.0.0',  # 监听所有网络接口
    'port': 5002,       # 服务端口
    'debug': False,     # 生产环境设为False
    'threaded': True,   # 启用多线程
    'max_content_length': 16 * 1024 * 1024  # 最大请求大小16MB
}

# 加载模型
# MODEL_PATH = "/share/project/lyx/openpi/checkpoints/pi0_agilex_orange/pi0_agilex_orange/50000"
MODEL_PATH = "/home/admin123/Desktop/90000"

# 全局模型变量
policy = None
to_tensor = transforms.ToTensor()

def load_policy():
    """加载openpi模型"""
    global policy
    try:
        logger.info("开始加载openpi模型...")
        # 使用openpi的配置和策略
        config = _config.get_config("pi0_agilex_orange_norm")
        policy = _policy_config.create_trained_policy(config, MODEL_PATH)        
        print(f"policy loaded.")    
        logger.info("openpi模型加载完成！")
        return True
    except Exception as e:
        logger.error(f"openpi模型加载失败: {e}")
        logger.error(traceback.format_exc())
        return False

def decode_image_base64(image_base64):
    """解码base64图片"""
    try:
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        image = to_tensor(image)
        return image
    except Exception as e:
        logger.error(f"图片解码失败: {e}")
        raise ValueError(f"图片解码失败: {e}")

def process_images(images_dict):
    """处理图片列表，适配openpi的图片格式"""
    try:
        sample_dict = {}
        # 根据openpi的配置调整图片键名
        for k in ['cam_high', 'cam_left_wrist', 'cam_right_wrist']:
            if k in images_dict:
                sample_dict[k] = decode_image_base64(images_dict[k])
            else:
                logger.warning(f"缺少图片: {k}")

    except Exception as e:
        logger.error(f"处理图片失败: {e}")
        raise ValueError(f"处理图片失败: {e}")

    return sample_dict

@app.route('/info', methods=['GET'])
def service_info():
    """服务信息端点"""
    return jsonify({
        "service_name": "OpenPI RoboBrain Robotics API",
        "version": "1.0.0",
        "endpoints": {
            "info": "/info", 
            "infer": "/infer"
        },
        "model_info": {
            "model_path": MODEL_PATH,
            "model_type": "openpi_pi0_agilex_orange_norm"
        },
        "timestamp": time.time()
    })

@app.route('/replay', methods=['GET'])
def replay_api():
    """ground truth qpos API"""
    try: 
        # with h5py.File('/share/project/yunfan/openpi/examples/agilex/replay.h5', 'r') as f:
        #     action = f['actions'][:]
        #     qpos = f['qpos'][:]

        # with open("",'r',encoding='utf-8') as f:
        #     data = json.load(f)
        
        # qpos = data['action']
        # qpos_real = data['action_real']
        # assert np.array(qpos).shape == (8, 50)
        # assert np.array(qpos).shape == (8, 50)

        qpos = np.load('/home/admin123/桌面/wjl/openpi/test_orange/actions_episode_000005.npy').tolist()
        # qpos_real = np.load('/share/project/chenghy/code/model_eval_real/8.13_pi0/50/true_action_G50_S4w.npy')
        eepose = []
        # qpos = []
        action_real = []

        
        # eepose = np.load("/share/project/lyx/robotics_final/action_h_30.npy").tolist()

        return jsonify({
            "success": True, 
            "eepose": eepose,
            "qpos": qpos,
            "action_real":action_real,
        })
    except Exception as e:
        logger.error(f"获取ground truth qpos失败: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/infer', methods=['POST'])
def infer_api():
    """推理API端点"""
    start_time = time.time()
    
    try:
        # 检查模型是否已加载
        if policy is None:
            print("400: 模型未加载，请检查服务状态")
            return jsonify({
                "success": False,
                "error": "模型未加载，请检查服务状态"
            }), 503
        
        # 解析请求数据
        data = request.get_json()
        if not data:
            print("400: 请求数据为空或格式错误!")
            return jsonify({
                "success": False,
                "error": "请求数据为空或格式错误"
            }), 400
        
        if 'eef_pose' not in data:
            print("400: 缺少必需字段: eef_pose!")
            return jsonify({
                "success": False,
                "error": "缺少必需字段: eef_pose"
            }), 400
        
        logging.info(f"received keys: {data}")
        
        images = data.get('images')
        state = data.get('state')  # 如果是 test_task_orange，就改成 state，否则用 qpos

        # 处理图片数据
        images_tensor = process_images(images)
        
        # 构建openpi格式的观察数据
        obs = {
            "images": {
                "cam_high": images_tensor['cam_high'],
                "cam_left_wrist": images_tensor['cam_left_wrist'],
                "cam_right_wrist": images_tensor['cam_right_wrist'],
            },
            "state": np.array(state[0]).astype(np.float32),
            "prompt": "task_orange_110_8.11",
        }
        logger.info(f"obs_images: {obs['images']['cam_left_wrist'].shape}")
        logger.info(f"obs_state: {obs['state'].shape}")
        logger.info(f"obs_prompt: {obs['prompt']}")
        logging.info(f"obs: {obs}")

        # 执行推理
        with torch.no_grad():
            output = policy.infer(obs)
            action = output['actions']

        # 添加处理时间信息
        processing_time = time.time() - start_time
        
        return jsonify({
            "success": True, 
            "qpos": action.tolist(), # (50, 14), target joint positions
            "processing_time": processing_time
        }), 200
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"推理失败: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": str(e),
            "processing_time": processing_time
        }), 500
    
# sid -> state
_sessions = {}  # {sid: {"obs": None, "actions": None, "ver": 0, "lock": threading.Lock(), "alive": True, "th": None}}
_model_lock = threading.Lock()  # 串行化访问模型，避免多线程并发进 policy.infer

def _parse_obs_from_json(data):
    """
    将客户端发来的 JSON 转为 policy.infer 的 obs。
    建议与你 /infer 里的构造方式完全一致（对齐训练预处理）：
    - 图像：建议统一到 224x224 + uint8(如需可在这里改)
    - state:float32
    - prompt:从 data 中取或给默认
    """
    images = data.get("images")
    state = data.get("state")
    if images is None or state is None:
        raise ValueError("missing images or state")

    images_tensor = process_images(images)  # 你已有的解码逻辑：base64->PIL->Tensor
    obs = {
        "images": {
            "cam_high": images_tensor.get("cam_high"),
            "cam_left_wrist": images_tensor.get("cam_left_wrist"),
            "cam_right_wrist": images_tensor.get("cam_right_wrist"),
        },
        "state": np.array(state[0]).astype(np.float32) if isinstance(state, list) else np.array(state).astype(np.float32),
        "prompt": data.get("instruction", "task_orange_110_8.11"),
    }
    return obs

def _ensure_worker(sid: str):
    s = _sessions[sid]
    if s.get("th") and s["th"].is_alive():
        return

    def _worker():
        while s["alive"]:
            # 拉最新 obs
            with s["lock"]:
                obs = s["obs"]
            if obs is None:
                time.sleep(0.005)
                continue
            try:
                # 串行访问模型
                with _model_lock:
                    out = policy.infer(obs)   # 一次性整段动作
                actions = out["actions"]
                with s["lock"]:
                    s["actions"] = actions
                    s["ver"] += 1
            except Exception as e:
                logger.error(f"[{sid}] worker infer error: {e}")
                time.sleep(0.02)

    s["alive"] = True
    s["th"] = threading.Thread(target=_worker, daemon=True)
    s["th"].start()

@app.route('/push_obs', methods=['POST'])
def push_obs():
    """
    客户端频繁上传最新 obs;快速返回（不阻塞在推理）。
    用法:POST /push_obs?sid=xxx   body=obs-json
    """
    sid = request.args.get("sid", "default")
    data = request.get_json()
    if sid not in _sessions:
        _sessions[sid] = {"obs": None, "actions": None, "ver": 0, "lock": threading.Lock(), "alive": True, "th": None}
    try:
        obs = _parse_obs_from_json(data)
        with _sessions[sid]["lock"]:
            _sessions[sid]["obs"] = obs
        _ensure_worker(sid)
        return ("", 204)  # 立即返回
    except Exception as e:
        logger.error(f"push_obs error: {e}")
        return jsonify({"success": False, "error": str(e)}), 400

@app.route('/pull_actions', methods=['GET'])
def pull_actions():
    """
    客户端长轮询拉最新动作块；有新版本就立刻返回。
    用法:GET /pull_actions?sid=xxx&after=VER&timeout_ms=5000
    """
    sid = request.args.get("sid", "default")
    after = int(request.args.get("after", "-1"))
    timeout_ms = int(request.args.get("timeout_ms", "5000"))
    t0 = time.time()

    while time.time() - t0 < timeout_ms / 1000.0:
        s = _sessions.get(sid)
        if s:
            with s["lock"]:
                ver = s["ver"]
                acts = s["actions"]
            if acts is not None and ver > after:
                # 注意：默认 JSON 可序列化；若是 ndarray，请转 list
                return jsonify({"ready": True, "ver": ver, "actions": np.asarray(acts).tolist()})
        time.sleep(0.005)

    s = _sessions.get(sid)
    cur_ver = s["ver"] if s else -1
    return jsonify({"ready": False, "ver": cur_ver})

@app.errorhandler(404)
def not_found(error):
    """404错误处理"""
    return jsonify({
        "success": False,
        "error": "接口不存在",
        "available_endpoints": ["/info", "/infer"]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """500错误处理"""
    return jsonify({
        "success": False,
        "error": "服务器内部错误"
    }), 500

if __name__ == '__main__':
    # 加载模型
    if not load_policy():
        logger.error("openpi模型加载失败，服务无法启动")
        sys.exit(1)
    
    print("load policy success")
    # 打印服务信息
    logger.info(f"OpenPI RoboBrain Robotics API 服务启动中...")
    logger.info(f"服务地址: http://{SERVICE_CONFIG['host']}:{SERVICE_CONFIG['port']}")
    logger.info(f"可用端点:")
    logger.info(f"  - GET  /info    - 服务信息")
    logger.info(f"  - POST /infer   - 推理接口")
    
    # 启动服务
    app.run(
        host=SERVICE_CONFIG['host'],
        port=SERVICE_CONFIG['port'],
        debug=SERVICE_CONFIG['debug'],
        threaded=SERVICE_CONFIG['threaded']
    ) 