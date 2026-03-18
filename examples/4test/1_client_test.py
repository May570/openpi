#!/usr/bin/env python3
"""
task_orange推理测试程序
从真实数据文件读取数据来测试OpenPI模型的推理功能
"""

import os
import sys
import json
import time
import numpy as np
import requests
import pandas as pd
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TaskTester:
    """task_orange推理测试器"""
    
    def __init__(self, service_url="http://127.0.0.1:5002", data_file="/home/admin123/Desktop/pi05_orange/episode_000003.parquet", test_interval=50, output_file="all_inference_results.json", predicted_actions_file="predicted_actions.npy", true_actions_file="true_actions.npy"):
        self.service_url = service_url
        self.data_file = data_file
        self.test_interval = test_interval  # 测试间隔（帧数）
        self.output_file = output_file  # 推理结果输出文件名
        self.predicted_actions_file = predicted_actions_file  # 预测action数组文件名
        self.true_actions_file = true_actions_file  # 真实action数组文件名
        self.test_data = None
        self.true_action = None
        self.df = None  # 保存整个数据框
        self.test_results = []  # 存储所有测试结果
        self.inference_results = []  # 存储所有推理结果
        self._load_real_data()
    
    def _load_real_data(self):
        """从parquet文件加载真实的task_orange数据"""
        try:
            logger.info(f"正在加载数据文件: {self.data_file}")
            
            if not os.path.exists(self.data_file):
                raise FileNotFoundError(f"数据文件不存在: {self.data_file}")
            
            # 读取parquet文件
            self.df = pd.read_parquet(self.data_file)
            logger.info(f"数据文件加载成功，共{len(self.df)}行数据")
            
            # 计算测试点
            self.test_points = self._calculate_test_points()
            logger.info(f"计算得到{len(self.test_points)}个测试点")
            
            # 构建第一个测试点的数据（用于初始化）
            first_row_data = self.df.iloc[self.test_points[0]]
            self.test_data = self._build_test_data_from_row(first_row_data, self.test_points[0])
            logger.info("测试数据构建完成")
            
        except Exception as e:
            logger.error(f"加载数据失败: {e}")
            raise
    
    def _build_test_data_from_row(self, row_data, row_index):
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
            self.true_action = self._extract_action_sequence(row_index)
            if self.true_action is not None:
                logger.info(f"提取真实action数据，维度: {len(self.true_action)} x {len(self.true_action[0]) if self.true_action else 0}")
            else:
                logger.warning("无法提取action序列，无法进行MSE对比")
            
            # 确保所有数据都是JSON可序列化的
            test_data = self._ensure_json_serializable(test_data)
            
            logger.info(f"构建的测试数据包含:")
            logger.info(f"  - qpos维度: {len(qpos)} x {len(qpos[0]) if qpos else 0}")
            logger.info(f"  - 图片数量: {len(images)}")
            
            return test_data
            
        except Exception as e:
            logger.error(f"构建测试数据失败: {e}")
            raise
    
    def _calculate_test_points(self):
        """计算测试点，每test_interval帧一个点"""
        try:
            total_frames = len(self.df)
            test_points = []
            
            # 从第0帧开始，每test_interval帧取一个点
            for i in range(0, total_frames, self.test_interval):
                # 确保有足够的数据进行action序列提取（需要50帧来匹配模型输出）
                if i + 50 <= total_frames:
                    test_points.append(i)
            
            logger.info(f"测试点: {test_points}")
            return test_points
            
        except Exception as e:
            logger.error(f"计算测试点失败: {e}")
            return [0]  # 默认至少测试第0帧
    
    def _extract_action_sequence(self, start_row_index):
        """从指定行开始提取连续的action序列"""
        try:
            if self.df is None:
                return None
            
            # 计算结束行索引（取50行或到文件末尾，匹配模型输出维度）
            end_row_index = min(start_row_index + 50, len(self.df))
            logger.info(f"提取action序列: 从第{start_row_index}行到第{end_row_index-1}行")
            
            # 提取连续的action数据
            action_sequence = []
            for i in range(start_row_index, end_row_index):
                row = self.df.iloc[i]
                if 'action' in row:
                    action_data = row['action']
                    # 检查action_data是否有效
                    if action_data is not None and (isinstance(action_data, np.ndarray) and action_data.size > 0) or (not isinstance(action_data, np.ndarray) and action_data):
                        if isinstance(action_data, np.ndarray):
                            action_sequence.append(action_data.tolist())
                        else:
                            action_sequence.append(action_data)
                    else:
                        logger.warning(f"第{i}行action数据为空")
                        break
                else:
                    logger.warning(f"第{i}行缺少action字段")
                    break
            
            if action_sequence:
                logger.info(f"成功提取{len(action_sequence)}个action")
                return action_sequence
            else:
                logger.warning("未提取到任何action数据")
                return None
                
        except Exception as e:
            logger.error(f"提取action序列失败: {e}")
            return None
    
    def _ensure_json_serializable(self, obj):
        """确保对象是JSON可序列化的"""
        if isinstance(obj, dict):
            return {key: self._ensure_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._ensure_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj
    
    def test_service_info(self):
        """测试服务信息接口"""
        try:
            logger.info("测试服务信息接口...")
            response = requests.get(f"{self.service_url}/info", timeout=10)
            
            if response.status_code == 200:
                info = response.json()
                logger.info("✅ 服务信息获取成功")
                logger.info(f"服务名称: {info.get('service_name')}")
                logger.info(f"版本: {info.get('version')}")
                logger.info(f"模型路径: {info.get('model_info', {}).get('model_path')}")
                return True
            else:
                logger.error(f"❌ 服务信息获取失败: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ 服务连接失败: {e}")
            return False
    
    def test_inference(self):
        """测试推理接口"""
        try:
            logger.info("测试推理接口...")
            logger.info(f"使用真实数据文件: {self.data_file}")
            
            # 对每个测试点进行推理
            for i, test_point in enumerate(self.test_points):
                logger.info(f"\n--- 测试点 {i+1}/{len(self.test_points)}: 第{test_point}帧 ---")
                
                # 构建当前测试点的数据
                row_data = self.df.iloc[test_point]
                test_data = self._build_test_data_from_row(row_data, test_point)
                
                if not test_data:
                    logger.error(f"❌ 测试点{test_point}数据构建失败")
                    continue
                
                # 执行推理
                start_time = time.time()
                response = requests.post(
                    f"{self.service_url}/infer",
                    json=test_data,
                    timeout=100
                )
                # response = requests.get(
                #     f"{self.service_url}/replay",
                #     json=test_data,
                #     timeout=100
                # )
                end_time = time.time()
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get('success'):
                        logger.info(f"✅ 测试点{test_point}推理成功")
                        logger.info(f"处理时间: {result.get('processing_time', 0):.3f}秒")
                        
                        # 计算MSE对比
                        actions = result.get('qpos', [])
                        if self.true_action and actions:
                            mse = self._calculate_mse(actions, self.true_action)
                            logger.info(f"与真实action的MSE: {mse:.6f}")
                            
                            # 可视化对比并保存到figs文件夹
                            self._visualize_comparison(actions, self.true_action, mse, test_point)
                            
                            # 保存测试结果
                            test_result = {
                                'test_point': test_point,
                                'mse': mse,
                                'processing_time': result.get('processing_time', 0),
                                'actions_shape': [len(actions), len(actions[0]) if actions else 0],
                                'visualization_file': f"figs/action_comparison_frame_{test_point:06d}.png"
                            }
                            self.test_results.append(test_result)
                        
                        # 保存推理结果到统一列表
                        inference_result = {
                            'test_point': test_point,
                            'predicted_actions': actions,  # 模型推理的动作序列
                            'true_actions': self.true_action,  # 真实的动作序列
                            'mse': mse if self.true_action and actions else None,
                            'actions_shape': [len(actions), len(actions[0]) if actions else 0] if actions else None,
                        }
                        self.inference_results.append(inference_result)
                        
                        # 同时保存单个推理结果文件（可选）
                        self._save_inference_result(result, f"inference_result_{test_point}.json")
                        
                    else:
                        logger.error(f"❌ 测试点{test_point}推理失败: {result.get('error')}")
                else:
                    logger.error(f"❌ 测试点{test_point}请求失败: {response.status_code}")
                    logger.error(f"错误信息: {response.text}")
                
                # 添加延迟避免请求过快
                time.sleep(1)
            
            # 生成综合测试报告
            if self.test_results:
                self._generate_test_report()
                
                # 保存所有推理结果
                self.save_all_inference_results()
                
                # 保存action数组
                self.save_actions_as_arrays()
                
                return True
            else:
                logger.error("❌ 所有测试点都失败了")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ 推理请求异常: {e}")
            return False
    
    def _save_inference_result(self, result, filename="inference_result.json"):
        """保存推理结果到文件"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ 推理结果已保存到: {filename}")
        except Exception as e:
            logger.error(f"❌ 保存推理结果失败: {e}")
    
    def save_all_inference_results(self, filename=None):
        """保存所有推理结果到一个统一的JSON文件"""
        try:
            if not self.inference_results:
                logger.warning("没有推理结果可保存")
                return
            
            # 使用实例变量中的文件名，如果没有指定的话
            if filename is None:
                filename = self.output_file
            
            # 构建完整的推理结果数据结构
            all_results = {
                'metadata': {
                    'service_url': self.service_url,
                    'data_file': self.data_file,
                    'test_interval': self.test_interval,
                    'total_test_points': len(self.test_points),
                    'successful_inferences': len(self.inference_results),
                    'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'test_points': self.test_points
                },
                'inference_results': self.inference_results
            }
            
            # 保存到文件
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ 所有推理结果已保存到: {filename}")
            logger.info(f"包含 {len(self.inference_results)} 个推理结果")
            
        except Exception as e:
            logger.error(f"❌ 保存所有推理结果失败: {e}")
    
    def save_actions_as_arrays(self, predicted_filename=None, true_filename=None):
        """将推理action和真实action保存为numpy数组格式 (检查点数量, 14, action chunk长度)"""
        try:
            if not self.inference_results:
                logger.warning("没有推理结果可保存为数组")
                return
            
            # 使用实例变量中的文件名，如果没有指定的话
            if predicted_filename is None:
                predicted_filename = self.predicted_actions_file
            if true_filename is None:
                true_filename = self.true_actions_file
            
            import numpy as np
            
            # 收集所有测试点的数据
            test_points = []
            predicted_actions_list = []
            true_actions_list = []
            
            for inference_result in self.inference_results:
                test_point = inference_result['test_point']
                predicted_actions = inference_result['predicted_actions']
                true_actions = inference_result['true_actions']
                
                if predicted_actions and true_actions:
                    test_points.append(test_point)
                    predicted_actions_list.append(predicted_actions)
                    true_actions_list.append(true_actions)
            
            if not predicted_actions_list:
                logger.warning("没有有效的action数据可保存")
                return
            
            # 转换为numpy数组
            predicted_array = np.array(predicted_actions_list)
            true_array = np.array(true_actions_list)
            
            # 检查数组形状
            logger.info(f"预测action数组形状: {predicted_array.shape}")
            logger.info(f"真实action数组形状: {true_array.shape}")
            
            # 保存为numpy数组文件
            np.save(predicted_filename, predicted_array)
            np.save(true_filename, true_array)
            
            logger.info(f"✅ 预测action已保存到: {predicted_filename}")
            logger.info(f"✅ 真实action已保存到: {true_filename}")
            
            # 同时保存为JSON格式（便于查看）
            actions_summary = {
                'metadata': {
                    'total_checkpoints': len(test_points),
                    'test_points': test_points,
                    'predicted_shape': predicted_array.shape.tolist(),
                    'true_shape': true_array.shape.tolist(),
                    'generated_at': time.strftime('%Y-%m-%d %H:%M:%S')
                },
                'predicted_actions': predicted_array.tolist(),
                'true_actions': true_array.tolist()
            }
            
            summary_filename = "actions_summary.json"
            with open(summary_filename, 'w', encoding='utf-8') as f:
                json.dump(actions_summary, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ 动作摘要已保存到: {summary_filename}")
            
        except Exception as e:
            logger.error(f"❌ 保存action数组失败: {e}")
    
    def _calculate_mse(self, predicted_actions, true_actions):
        """计算预测action与真实action的MSE"""
        try:
            import numpy as np
            
            # 转换为numpy数组
            pred = np.array(predicted_actions)
            true = np.array(true_actions)
            
            # 确保维度匹配
            if pred.shape != true.shape:
                logger.info(f"维度调整: 预测{pred.shape} vs 真实{true.shape}")
                # 取较小的维度进行对比
                min_steps = min(pred.shape[0], true.shape[0])
                pred = pred[:min_steps]
                true = true[:min_steps]
                logger.info(f"调整为共同维度: {min_steps} x {pred.shape[1] if pred.shape[1:] else 0}")
            
            # 计算MSE
            mse = np.mean((pred - true) ** 2)
            return mse
            
        except Exception as e:
            logger.error(f"计算MSE失败: {e}")
            return None
    
    def _visualize_comparison(self, predicted_actions, true_actions, mse, test_point):
        """可视化预测action与真实action的对比 - 每个自由度一张子图"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # 转换为numpy数组
            pred = np.array(predicted_actions)
            true = np.array(true_actions)
            
            # 确保维度匹配
            min_steps = min(pred.shape[0], true.shape[0])
            pred = pred[:min_steps]
            true = true[:min_steps]
            logger.info(f"可视化维度: {min_steps} x {pred.shape[1] if pred.shape[1:] else 0}")
            
            # 计算子图布局（每行最多4个子图）
            num_joints = pred.shape[1]
            cols = min(4, num_joints)
            rows = (num_joints + cols - 1) // cols
            
            # 创建图表
            fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
            fig.suptitle(f'Joint Trajectory Comparison - Frame {test_point} (MSE: {mse:.6f})', fontsize=16)
            
            # 确保axes是二维数组
            if rows == 1:
                axes = axes.reshape(1, -1)
            if cols == 1:
                axes = axes.reshape(-1, 1)
            
            # 为每个自由度创建子图
            for i in range(num_joints):
                row = i // cols
                col = i % cols
                ax = axes[row, col]
                
                # 绘制预测和真实轨迹
                ax.plot(pred[:, i], label='Predicted', color='blue', linewidth=2, alpha=0.8)
                ax.plot(true[:, i], label='Ground Truth', color='red', linestyle='--', linewidth=2, alpha=0.8)
                
                ax.set_title(f'Joint {i+1}')
                ax.set_xlabel('Time Steps')
                ax.set_ylabel('Joint Angle')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # 隐藏多余的子图
            for i in range(num_joints, rows * cols):
                row = i // cols
                col = i % cols
                axes[row, col].set_visible(False)
            
            plt.tight_layout()
            
            # 创建figs文件夹
            import os
            figs_dir = "figs"
            if not os.path.exists(figs_dir):
                os.makedirs(figs_dir)
                logger.info(f"✅ 创建figs文件夹: {figs_dir}")
            
            # 保存图表到figs文件夹
            fig_filename = f"action_comparison_frame_{test_point:06d}.png"
            fig_path = os.path.join(figs_dir, fig_filename)
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            logger.info(f"✅ 可视化图表已保存到: {fig_path}")
            
            # Display chart (if in environment with display)
            try:
                plt.show()
            except:
                logger.info("Cannot display chart, saved to file")
            
        except ImportError:
            logger.warning("matplotlib not installed, skipping visualization")
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
    
    def _generate_test_report(self):
        """生成综合测试报告"""
        try:
            if not self.test_results:
                logger.warning("没有测试结果可生成报告")
                return
            
            # 计算统计信息
            mse_values = [result['mse'] for result in self.test_results if result['mse'] is not None]
            processing_times = [result['processing_time'] for result in self.test_results]
            
            report = {
                'test_summary': {
                    'total_test_points': len(self.test_points),
                    'successful_tests': len(self.test_results),
                    'test_interval': self.test_interval,
                    'test_points': self.test_points
                },
                'performance_metrics': {
                    'mse_statistics': {
                        'mean_mse': np.mean(mse_values) if mse_values else None,
                        'std_mse': np.std(mse_values) if mse_values else None,
                        'min_mse': np.min(mse_values) if mse_values else None,
                        'max_mse': np.max(mse_values) if mse_values else None
                    },
                    'processing_time_statistics': {
                        'mean_time': np.mean(processing_times) if processing_times else None,
                        'std_time': np.std(processing_times) if processing_times else None,
                        'min_time': np.min(processing_times) if processing_times else None,
                        'max_time': np.max(processing_times) if processing_times else None
                    }
                },
                'detailed_results': self.test_results
            }
            
            # 保存报告
            report_file = "test_report.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ 测试报告已保存到: {report_file}")
            
            # 打印摘要
            logger.info("\n" + "="*50)
            logger.info("测试报告摘要")
            logger.info("="*50)
            logger.info(f"总测试点: {len(self.test_points)}")
            logger.info(f"成功测试: {len(self.test_results)}")
            logger.info(f"测试间隔: {self.test_interval}帧")
            if mse_values:
                logger.info(f"平均MSE: {np.mean(mse_values):.6f}")
                logger.info(f"MSE标准差: {np.std(mse_values):.6f}")
            if processing_times:
                logger.info(f"平均处理时间: {np.mean(processing_times):.3f}秒")
            logger.info(f"可视化图表: {len(self.test_results)}个，保存在figs/文件夹")
            
            # 打印动作对比摘要
            logger.info("\n动作对比摘要:")
            for i, test_result in enumerate(self.test_results):
                test_point = test_result['test_point']
                mse = test_result['mse']
                actions_shape = test_result['actions_shape']
                logger.info(f"  测试点{test_point}: MSE={mse:.6f}, 动作形状={actions_shape}")
            
            logger.info("="*50)
            
        except Exception as e:
            logger.error(f"生成测试报告失败: {e}")
    
    def run_full_test(self):
        """运行完整测试"""
        logger.info("🚀 开始task_orange推理测试...")
        logger.info(f"测试服务地址: {self.service_url}")
        logger.info(f"数据文件: {self.data_file}")
        
        # 测试1: 服务信息
        if not self.test_service_info():
            logger.error("❌ 服务信息测试失败，请检查服务是否启动")
            return False
        
        # 测试2: 推理功能
        if not self.test_inference():
            logger.error("❌ 推理测试失败")
            return False
        
        logger.info("✅ 所有测试通过！")
        return True
    
    def save_test_data(self, filename="test_task_orange_data.json"):
        """保存测试数据到文件"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.test_data, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ 测试数据已保存到: {filename}")
        except Exception as e:
            logger.error(f"❌ 保存测试数据失败: {e}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="task_orange推理测试程序")
    parser.add_argument(
        "--url", 
        default="http://127.0.0.1:5001",
        help="服务地址 (默认: http://127.0.0.1:5001)"
    )
    parser.add_argument(
        "--data-file",
        default="/share/project/lyx/RoboTwin/lerobot_data/huggingface/lerobot/open_laptop-demo_clean-50/data/chunk-000/episode_000001.parquet",
        help="数据文件路径"
    )
    parser.add_argument(
        "--save-data",
        action="store_true",
        help="保存测试数据到文件"
    )
    parser.add_argument(
        "--test-interval",
        type=int,
        default=50,
        help="测试间隔（帧数，默认: 50）"
    )
    parser.add_argument(
        "--output-file",
        default="all_inference_results.json",
        help="推理结果输出文件名 (默认: all_inference_results.json)"
    )
    parser.add_argument(
        "--predicted-actions-file",
        default="predicted_actions.npy",
        help="预测action数组文件名 (默认: predicted_actions.npy)"
    )
    parser.add_argument(
        "--true-actions-file",
        default="true_actions.npy",
        help="真实action数组文件名 (默认: true_actions.npy)"
    )
    
    args = parser.parse_args()
    
    # 创建测试器
    tester = TaskTester(args.url, args.data_file, args.test_interval, args.output_file, args.predicted_actions_file, args.true_actions_file)
    
    # 运行测试
    success = tester.run_full_test()
    
    # 保存测试数据（如果需要）
    if args.save_data:
        tester.save_test_data()
    
    # 退出码
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
