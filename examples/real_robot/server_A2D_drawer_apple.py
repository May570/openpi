"""
A2D 双臂机器人推理服务 - Joint 控制模式

=== 维度信息 ===

输入 joint_state: 16维
  右臂 (8维): joint1-7 + gripper
  左臂 (8维): joint1-7 + gripper
  顺序: [右臂关节(7) + 右爪(1) + 左臂关节(7) + 左爪(1)]

输出 joint: (30, 16) - 30步动作序列，每步16维 (模型直接输出绝对位置)

图像输入:
  cam_head: 头部相机 (320x240)
  cam_left_wrist: 左腕相机 (320x240)
  cam_right_wrist: 右腕相机 (320x240)

=== POST /infer 输入样例 ===
{
  "joint_state": [rj1, rj2, rj3, rj4, rj5, rj6, rj7, rg, lj1, lj2, lj3, lj4, lj5, lj6, lj7, lg],  # shape: [16]
  "instruction": "pick up the blackboard eraser and erase the handwriting",
  "images": {
    "cam_head": "<base64字符串>",
    "cam_left_wrist": "<base64字符串>",
    "cam_right_wrist": "<base64字符串>"
  }
}

=== POST /infer 输出样例 ===
{
  "success": true,
  "joint": [[...], [...], ...],  # shape: (30, 16)
  "processing_time": 0.123
}
"""

import os
import io
import base64
from PIL import Image
import sys
import torch
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import time
import traceback
from torchvision import transforms
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 图像日志目录
IMAGE_LOG_DIR = "./image_log"
os.makedirs(IMAGE_LOG_DIR, exist_ok=True)

app = Flask(__name__)
CORS(app)

# 服务配置
SERVICE_CONFIG = {
    'host': '0.0.0.0',
    'port': 5004,
    'debug': False,
    'threaded': True,
    'max_content_length': 16 * 1024 * 1024
}

# 模型路径
MODEL_PATH = "/share/project/wujiling/models/finetune/pi05_a2d_drawer_apple/35000"

# 全局变量
policy = None
to_tensor = transforms.ToTensor()


def load_policy():
    """加载openpi模型"""
    global policy
    try:
        logger.info("开始加载openpi模型...")
        config = _config.get_config("pi05_a2d_drawer_apple_paligemma_init")
        policy = _policy_config.create_trained_policy(config, MODEL_PATH)
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


def decode_image_base64_to_pil(image_base64):
    """解码base64图片为PIL Image"""
    try:
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        return image
    except Exception as e:
        logger.error(f"图片解码失败: {e}")
        raise ValueError(f"图片解码失败: {e}")


def save_images_log(images_dict):
    """保存图像日志用于调试"""
    try:
        key_mapping = {
            'cam_head': 'cam_head',
            'cam_left_wrist': 'cam_left_wrist',
            'cam_right_wrist': 'cam_right_wrist'
        }
        for client_key, log_name in key_mapping.items():
            if client_key in images_dict and images_dict[client_key] is not None:
                pil_img = decode_image_base64_to_pil(images_dict[client_key])
                log_path = os.path.join(IMAGE_LOG_DIR, f"{log_name}.png")
                pil_img.save(log_path)
    except Exception as e:
        logger.warning(f"保存图像日志失败: {e}")


def process_images(images_dict):
    """处理图片，适配openpi格式"""
    try:
        sample_dict = {}
        key_mapping = {
            'cam_head': 'cam_high',
            'cam_left_wrist': 'cam_left_wrist',
            'cam_right_wrist': 'cam_right_wrist'
        }
        for client_key, model_key in key_mapping.items():
            if client_key in images_dict and images_dict[client_key] is not None:
                sample_dict[model_key] = decode_image_base64(images_dict[client_key])
            else:
                logger.warning(f"缺少图片: {client_key}")
        return sample_dict
    except Exception as e:
        logger.error(f"处理图片失败: {e}")
        raise ValueError(f"处理图片失败: {e}")


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查"""
    try:
        if policy is None:
            return jsonify({"status": "error", "message": "模型未加载"}), 503
        
        gpu_info = {}
        if torch.cuda.is_available():
            gpu_info = {
                "available": True,
                "device_name": torch.cuda.get_device_name(),
                "memory_allocated": f"{torch.cuda.memory_allocated() / 1024**3:.2f}GB"
            }
        else:
            gpu_info = {"available": False}
        
        return jsonify({"status": "healthy", "model_loaded": True, "gpu_info": gpu_info})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/info', methods=['GET'])
def service_info():
    """服务信息"""
    return jsonify({
        "service_name": "A2D Joint Control API",
        "version": "1.0.0",
        "control_mode": "joint",
        "endpoints": {
            "health": "/health",
            "info": "/info",
            "infer": "/infer"
        },
        "input_dim": 16,
        "output_shape": "(30, 16)",
        "timestamp": time.time()
    })


@app.route('/infer', methods=['POST'])
def infer_api():
    """推理API - Joint控制模式"""
    start_time = time.time()
    
    try:
        if policy is None:
            return jsonify({"success": False, "error": "模型未加载"}), 503
        
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "请求数据为空"}), 400
        
        if 'joint_state' not in data:
            return jsonify({"success": False, "error": "缺少必需字段: joint_state"}), 400
        
        # 获取数据
        images = data.get('images')
        joint_state = data.get('joint_state')  # 16维: 右臂7+右爪1+左臂7+左爪1
        instruction = data.get('instruction', '')
        
        # 处理图片
        images_tensor = process_images(images)
        save_images_log(images)
        
        # 转换 joint_state
        state = np.array(joint_state).astype(np.float32)
        logger.info(f"joint_state 维度: {state.shape}")
        
        # 构建观察数据
        obs = {
            "images": {
                "cam_high": images_tensor.get('cam_high'),
                "cam_left_wrist": images_tensor.get('cam_left_wrist'),
                "cam_right_wrist": images_tensor.get('cam_right_wrist'),
            },
            "state": state,
            "prompt": instruction if instruction else "task_orange_110_8.11",
        }
        
        logger.info(f"obs_state: {obs['state'].shape}")
        logger.info(f"obs_prompt: {obs['prompt']}")
        
        # 执行推理 - 模型直接输出绝对位置
        with torch.no_grad():
            output = policy.infer(obs)
            actions = output['actions']  # (30, 16) 绝对关节位置
        
        processing_time = time.time() - start_time
        logger.info(f"输出 joint shape: {np.array(actions).shape}")
        
        return jsonify({
            "success": True,
            "joint": actions.tolist(),  # 直接返回
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


@app.errorhandler(404)
def not_found(error):
    return jsonify({"success": False, "error": "接口不存在"}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"success": False, "error": "服务器内部错误"}), 500


if __name__ == '__main__':
    # 加载模型
    if not load_policy():
        logger.error("模型加载失败，服务无法启动")
        sys.exit(1)
    
    logger.info("模型加载成功")
    logger.info(f"A2D Joint Control API 服务启动中...")
    logger.info(f"服务地址: http://{SERVICE_CONFIG['host']}:{SERVICE_CONFIG['port']}")
    logger.info(f"控制模式: Joint (绝对位置)")
    logger.info(f"输入维度: 16 (右臂7+右爪1+左臂7+左爪1)")
    logger.info(f"输出维度: (30, 16)")
    
    app.run(
        host=SERVICE_CONFIG['host'],
        port=SERVICE_CONFIG['port'],
        debug=SERVICE_CONFIG['debug'],
        threaded=SERVICE_CONFIG['threaded']
    )
