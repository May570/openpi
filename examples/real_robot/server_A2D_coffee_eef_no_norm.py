"""
A2D 双臂机器人推理服务 - EEF 控制模式 (四元数 ↔ 欧拉角转换)

=== 维度信息 ===

客户端输入 eef_pose: 16维 (四元数格式)
  右臂 (8维): xyz(3) + quaternion(4) + gripper(1)
  左臂 (8维): xyz(3) + quaternion(4) + gripper(1)
  顺序: [右臂位置(3) + 右臂四元数(4) + 右爪(1) + 左臂位置(3) + 左臂四元数(4) + 左爪(1)]

模型内部: 14维 (欧拉角格式)
  右臂 (7维): xyz(3) + euler(3) + gripper(1)
  左臂 (7维): xyz(3) + euler(3) + gripper(1)
  
服务端输出 eepose: (30, 16) - 30步动作序列，每步16维 (四元数格式)

转换流程:
  客户端16维(四元数) -> 服务端转换为14维(欧拉角) -> 模型推理 -> 
  服务端转换为16维(四元数) -> 返回客户端

图像输入:
  cam_head: 头部相机 (320x240)
  cam_left_wrist: 左腕相机 (320x240)
  cam_right_wrist: 右腕相机 (320x240)

=== POST /infer 输入样例 ===
{
  "eef_pose": [rx, ry, rz, rqx, rqy, rqz, rqw, rg, lx, ly, lz, lqx, lqy, lqz, lqw, lg],  # shape: [16]
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
  "eepose": [[...], [...], ...],  # shape: (30, 16)
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
from scipy.spatial.transform import Rotation as R

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

# ========== 模型配置 ==========
# 每次启动只需修改以下两项：
MODEL_PATH = "/share/project/wujiling/openpi/checkpoints/pi05_A2D_pour_eef/pi05_A2D_pour_eef/30000"
MODEL_CONFIG_NAME = "pi05_A2D_pour_eef"
# ==============================

# 全局变量
policy = None
to_tensor = transforms.ToTensor()


def load_policy():
    """加载openpi模型"""
    global policy
    try:
        logger.info("开始加载openpi模型...")
        logger.info(f"模型路径: {MODEL_PATH}")
        logger.info(f"配置名称: {MODEL_CONFIG_NAME}")
        config = _config.get_config(MODEL_CONFIG_NAME)
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


def quaternion_to_euler(qx, qy, qz, qw):
    """
    将四元数转换为欧拉角(roll, pitch, yaw)
    
    参数:
        qx, qy, qz, qw: 四元数分量
    
    返回:
        欧拉角 [roll, pitch, yaw] (弧度)
    """
    r = R.from_quat([qx, qy, qz, qw])
    return r.as_euler('xyz', degrees=False)


def euler_to_quaternion(roll, pitch, yaw):
    """
    将欧拉角转换为四元数
    
    参数:
        roll, pitch, yaw: 欧拉角 (弧度)
    
    返回:
        四元数 [qx, qy, qz, qw]
    """
    r = R.from_euler('xyz', [roll, pitch, yaw], degrees=False)
    return r.as_quat()


def convert_eef_16d_to_14d(eef_pose_16d):
    """
    将16维EEF位姿(四元数)转换为14维EEF位姿(欧拉角)
    
    输入 (16维):
        [右xyz(3) + 右quat(4) + 右爪(1) + 左xyz(3) + 左quat(4) + 左爪(1)]
    
    输出 (14维):
        [右xyz(3) + 右euler(3) + 右爪(1) + 左xyz(3) + 左euler(3) + 左爪(1)]
    """
    eef_pose_16d = np.array(eef_pose_16d).astype(np.float32)
    
    # 提取右臂
    right_xyz = eef_pose_16d[0:3]
    right_quat = eef_pose_16d[3:7]  # [qx, qy, qz, qw]
    right_gripper = eef_pose_16d[7]
    
    # 提取左臂
    left_xyz = eef_pose_16d[8:11]
    left_quat = eef_pose_16d[11:15]  # [qx, qy, qz, qw]
    left_gripper = eef_pose_16d[15]
    
    # 转换四元数为欧拉角
    right_euler = quaternion_to_euler(*right_quat)  # [roll, pitch, yaw]
    left_euler = quaternion_to_euler(*left_quat)
    
    # 组合为14维
    eef_pose_14d = np.concatenate([
        right_xyz,      # 3
        right_euler,    # 3
        [right_gripper],# 1
        left_xyz,       # 3
        left_euler,     # 3
        [left_gripper]  # 1
    ])
    
    logger.info(f"转换: 16维(四元数) -> 14维(欧拉角)")
    logger.info(f"  右臂四元数: {right_quat} -> 欧拉角: {right_euler}")
    logger.info(f"  左臂四元数: {left_quat} -> 欧拉角: {left_euler}")
    
    return eef_pose_14d


def convert_eef_14d_to_16d(eef_pose_14d):
    """
    将14维EEF位姿(欧拉角)转换为16维EEF位姿(四元数)
    
    输入 (14维):
        [右xyz(3) + 右euler(3) + 右爪(1) + 左xyz(3) + 左euler(3) + 左爪(1)]
    
    输出 (16维):
        [右xyz(3) + 右quat(4) + 右爪(1) + 左xyz(3) + 左quat(4) + 左爪(1)]
    """
    eef_pose_14d = np.array(eef_pose_14d).astype(np.float32)
    
    # 提取右臂
    right_xyz = eef_pose_14d[0:3]
    right_euler = eef_pose_14d[3:6]  # [roll, pitch, yaw]
    right_gripper = eef_pose_14d[6]
    
    # 提取左臂
    left_xyz = eef_pose_14d[7:10]
    left_euler = eef_pose_14d[10:13]  # [roll, pitch, yaw]
    left_gripper = eef_pose_14d[13]
    
    # 转换欧拉角为四元数
    right_quat = euler_to_quaternion(*right_euler)  # [qx, qy, qz, qw]
    left_quat = euler_to_quaternion(*left_euler)
    
    # 组合为16维
    eef_pose_16d = np.concatenate([
        right_xyz,      # 3
        right_quat,     # 4
        [right_gripper],# 1
        left_xyz,       # 3
        left_quat,      # 4
        [left_gripper]  # 1
    ])
    
    return eef_pose_16d


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
        "service_name": "A2D EEF Control API",
        "version": "2.0.0",
        "control_mode": "eepose",
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
    """推理API - EEF控制模式"""
    start_time = time.time()
    
    try:
        if policy is None:
            return jsonify({"success": False, "error": "模型未加载"}), 503
        
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "请求数据为空"}), 400
        
        if 'eef_pose' not in data:
            return jsonify({"success": False, "error": "缺少必需字段: eef_pose"}), 400
        
        # 获取数据
        images = data.get('images')
        eef_pose = data.get('eef_pose')  # 16维: 右臂xyz+quat+gripper + 左臂xyz+quat+gripper
        instruction = data.get('instruction', '')
        
        # 处理图片
        images_tensor = process_images(images)
        save_images_log(images)
        
        # 转换 eef_pose: 16维(四元数) -> 14维(欧拉角)
        logger.info(f"接收到 eef_pose 维度: {len(eef_pose)}")
        state_14d = convert_eef_16d_to_14d(eef_pose)
        logger.info(f"转换后 state 维度: {state_14d.shape}")
        
        # 转换为 torch tensor
        state_tensor = torch.from_numpy(state_14d).float()
        
        # 构建观察数据
        # 同时提供 state 和 eef_state，因为 norm_stats 可能使用 eef_state 键
        obs = {
            "images": {
                "cam_high": images_tensor.get('cam_high'),
                "cam_left_wrist": images_tensor.get('cam_left_wrist'),
                "cam_right_wrist": images_tensor.get('cam_right_wrist'),
            },
            "state": state_tensor,      # 14维欧拉角格式
            "eef_state": state_tensor,  # 同样的数据，用于 norm_stats
            "prompt": instruction if instruction else "task_orange_110_8.11",
        }
        
        logger.info(f"obs['state'] shape: {obs['state'].shape}")
        logger.info(f"obs['state'] dtype: {obs['state'].dtype}")
        logger.info(f"obs['prompt']: {obs['prompt']}")
        
        # 手动执行推理，绕过有问题的 Unnormalize transform
        with torch.no_grad():
            # 1. 应用 input_transform
            inputs = policy._input_transform(obs)
            
            # 2. 准备模型输入并推理
            from openpi.models import model as _model
            import jax.numpy as jnp
            
            # 检查是否是 PyTorch 模型
            is_pytorch = hasattr(policy, '_is_pytorch_model') and policy._is_pytorch_model
            
            if is_pytorch:
                # PyTorch 模型处理
                inputs_tensor = {}
                for k, v in inputs.items():
                    if isinstance(v, dict):
                        inputs_tensor[k] = {}
                        for kk, vv in v.items():
                            if isinstance(vv, torch.Tensor):
                                inputs_tensor[k][kk] = vv.to(policy._pytorch_device)[None, ...]
                            else:
                                inputs_tensor[k][kk] = torch.from_numpy(np.array(vv)).to(policy._pytorch_device)[None, ...]
                    else:
                        if isinstance(v, torch.Tensor):
                            inputs_tensor[k] = v.to(policy._pytorch_device)[None, ...]
                        else:
                            inputs_tensor[k] = torch.from_numpy(np.array(v)).to(policy._pytorch_device)[None, ...]
                
                observation = _model.Observation.from_dict(inputs_tensor)
                actions = policy._sample_actions(policy._pytorch_device, observation)
                actions_np = actions[0, ...].detach().cpu().numpy()
                state_np = inputs["state"] if isinstance(inputs["state"], np.ndarray) else inputs["state"].cpu().numpy()
            else:
                # JAX 模型处理 - 转换为 JAX Array
                inputs_jax = {}
                for k, v in inputs.items():
                    if isinstance(v, dict):
                        inputs_jax[k] = {}
                        for kk, vv in v.items():
                            if isinstance(vv, torch.Tensor):
                                inputs_jax[k][kk] = jnp.asarray(vv.cpu().numpy())[np.newaxis, ...]
                            else:
                                inputs_jax[k][kk] = jnp.asarray(np.array(vv))[np.newaxis, ...]
                    else:
                        if isinstance(v, torch.Tensor):
                            inputs_jax[k] = jnp.asarray(v.cpu().numpy())[np.newaxis, ...]
                        else:
                            inputs_jax[k] = jnp.asarray(np.array(v))[np.newaxis, ...]
                
                observation = _model.Observation.from_dict(inputs_jax)
                
                # JAX 模型需要 RNG key
                import jax
                if not hasattr(policy, '_rng'):
                    policy._rng = jax.random.key(0)
                policy._rng, sample_rng = jax.random.split(policy._rng)
                
                actions = policy._sample_actions(sample_rng, observation)
                actions_np = np.asarray(actions[0, ...])
                state_np = inputs["state"] if isinstance(inputs["state"], np.ndarray) else np.asarray(inputs["state"])
            
            # 3. 手动应用 output_transform（跳过 Unnormalize）
            # 3a. AlohaOutputs: 只取前14维
            from openpi.policies import aloha_policy
            actions_np = actions_np[:, :14]  # 确保只有14维
            
            logger.info(f"模型原始输出 (前3步): {actions_np[:3]}")
            logger.info(f"模型原始输出范围: min={actions_np.min()}, max={actions_np.max()}")
            
            # 暂时跳过坐标转换和 delta 累加，直接使用模型输出
            # actions_np = aloha_policy._encode_actions(actions_np, adapt_to_pi=True)
            # from openpi.transforms import make_bool_mask
            # delta_mask = make_bool_mask(6, -1, 6, -1)
            # mask_array = np.asarray(delta_mask)
            # dims = mask_array.shape[-1]
            # actions_np[:, :dims] += np.expand_dims(np.where(mask_array, state_np[:dims], 0), axis=0)
            
            logger.info(f"最终输出 (前3步): {actions_np[:3]}")
            
            actions_14d = actions_np
        
        logger.info(f"模型输出 shape: {actions_14d.shape}")
        
        # 转换输出: 14维(欧拉角) -> 16维(四元数)
        actions_16d = []
        for action_14d in actions_14d:
            action_16d = convert_eef_14d_to_16d(action_14d)
            actions_16d.append(action_16d.tolist())
        
        processing_time = time.time() - start_time
        logger.info(f"转换后输出 eepose shape: ({len(actions_16d)}, {len(actions_16d[0])})")
        
        return jsonify({
            "success": True,
            "eepose": actions_16d,  # (30, 16) 四元数格式
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
    logger.info(f"A2D EEF Control API 服务启动中...")
    logger.info(f"服务地址: http://{SERVICE_CONFIG['host']}:{SERVICE_CONFIG['port']}")
    logger.info(f"控制模式: EEF (绝对位姿)")
    logger.info(f"输入维度: 16 (右臂xyz+quat+gripper + 左臂xyz+quat+gripper)")
    logger.info(f"输出维度: (30, 16)")
    logger.info(f"注意: 新版本模型已内置norm处理，无需额外归一化")
    
    app.run(
        host=SERVICE_CONFIG['host'],
        port=SERVICE_CONFIG['port'],
        debug=SERVICE_CONFIG['debug'],
        threaded=SERVICE_CONFIG['threaded']
    )
