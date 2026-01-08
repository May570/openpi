"""
XHT 双臂机器人推理服务 - Joint 控制模式

=== 维度信息 ===

输入 state (qpos): 14维
  右臂 (7维): joint1-6 + gripper
  左臂 (7维): joint1-6 + gripper
  顺序: [右臂关节(6) + 右爪(1) + 左臂关节(6) + 左爪(1)]

输出 qpos: (N, 14) - N步动作序列，每步14维

图像输入:
  cam_head: 头部相机 (image_right)
  cam_left_wrist: 左腕相机
  cam_right_wrist: 右腕相机

=== POST /infer 输入样例 ===
{
  "state": [rj1, rj2, rj3, rj4, rj5, rj6, rg, lj1, lj2, lj3, lj4, lj5, lj6, lg],  # shape: [14]
  "eef_pose": [...],  # shape: [16] (可选，用于eef模式)
  "instruction": "spell baai",
  "images": {
    "cam_head": "<base64字符串>",
    "cam_left_wrist": "<base64字符串>",
    "cam_right_wrist": "<base64字符串>"
  }
}

=== POST /infer 输出样例 ===
{
  "success": true,
  "qpos": [[...], [...], ...],  # shape: (N, 14) - joint模式
  "eepose": [[...], [...], ...],  # shape: (N, 16) - eef模式
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
    'port': 5001,
    'debug': False,
    'threaded': True,
    'max_content_length': 16 * 1024 * 1024
}

# 模型路径和配置名 - 根据实际情况修改
MODEL_PATH = "/share/project/wujiling/models/finetune/pi05_spell_baai/30000"
CONFIG_NAME = "pi05_spell_baai"

# 全局变量
policy = None
to_tensor = transforms.ToTensor()


def load_policy():
    """加载openpi模型"""
    global policy
    try:
        logger.info("开始加载openpi模型...")
        config = _config.get_config(CONFIG_NAME)
        policy = _policy_config.create_trained_policy(config, MODEL_PATH)
        logger.info("openpi模型加载完成！")
        return True
    except Exception as e:
        logger.error(f"openpi模型加载失败: {e}")
        logger.error(traceback.format_exc())
        return False


def decode_image_base64(image_base64):
    """解码base64图片为tensor"""
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
    """处理图片，适配模型格式"""
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
        "service_name": "XHT Joint Control API",
        "version": "1.0.0",
        "control_mode": "joint",
        "endpoints": {
            "health": "/health",
            "info": "/info",
            "infer": "/infer"
        },
        "input_dim": 14,
        "output_shape": "(N, 14)",
        "timestamp": time.time()
    })


@app.route('/infer', methods=['POST'])
def infer_api():
    """
    推理API - 支持 Joint 和 EEF 控制模式
    
    输入:
      - state (qpos): 14维 [右臂6+右爪1+左臂6+左爪1]
      - eef_pose: 16维 (可选)
      - instruction: 任务指令
      - images: 图像字典
    
    输出:
      - qpos: (N, 14) joint模式动作
      - eepose: (N, 16) eef模式动作
    """
    start_time = time.time()
    
    try:
        if policy is None:
            return jsonify({"success": False, "errowerkzeugr": "模型未加载"}), 503
        
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "请求数据为空"}), 400
        
        # 获取数据
        images = data.get('images')
        state = data.get('state')  # 14维: 右臂6+右爪1+左臂6+左爪1
        eef_pose = data.get('eef_pose')  # 16维 (可选)
        instruction = data.get('instruction', '')
        
        if state is None:
            return jsonify({"success": False, "error": "缺少必需字段: state"}), 400
        
        # 处理图片
        images_tensor = process_images(images)
        save_images_log(images)
        
        # 转换 state
        state_arr = np.array(state).astype(np.float32)
        logger.info(f"state 维度: {state_arr.shape}")
        logger.info(f"instruction: {instruction}")
        
        # 构建观察数据
        obs = {
            "images": {
                "cam_high": images_tensor.get('cam_high'),
                "cam_left_wrist": images_tensor.get('cam_left_wrist'),
                "cam_right_wrist": images_tensor.get('cam_right_wrist'),
            },
            "state": state_arr,
            "prompt": instruction if instruction else "default_task",
        }
        
        if eef_pose is not None:
            obs["eef_pose"] = np.array(eef_pose).astype(np.float32)
        
        # 执行推理
        with torch.no_grad():
            output = policy.infer(obs)
            qpos_actions = output['actions']  # (N, 14) 关节位置
        
        processing_time = time.time() - start_time
        logger.info(f"输出 qpos shape: {np.array(qpos_actions).shape}")
        
        return jsonify({
            "success": True,
            "qpos": qpos_actions.tolist(),  # (N, 14) joint模式
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
    logger.info(f"XHT Joint Control API 服务启动中...")
    logger.info(f"服务地址: http://{SERVICE_CONFIG['host']}:{SERVICE_CONFIG['port']}")
    logger.info(f"控制模式: Joint")
    logger.info(f"输入维度: 14 (右臂6+右爪1+左臂6+左爪1)")
    logger.info(f"输出维度: (N, 14)")
    
    app.run(
        host=SERVICE_CONFIG['host'],
        port=SERVICE_CONFIG['port'],
        debug=SERVICE_CONFIG['debug'],
        threaded=SERVICE_CONFIG['threaded']
    )
