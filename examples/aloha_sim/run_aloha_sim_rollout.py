import pathlib
import time
import logging
from datetime import datetime

# ✅ 项目内部导入（注意 packages 和 src 结构）
from src.openpi.policies import policy as _policy
from src.openpi.policies import policy_config as _policy_config
from src.openpi.training import config as _config

from openpi_client.action_chunk_broker import ActionChunkBroker
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent

# 当前目录内
import env as _env
import saver as _saver


def load_policy_from_cache(config_name: str, cache_dir: str) -> _policy.Policy:
    cfg = _config.get_config(config_name)
    return _policy_config.create_trained_policy(cfg, cache_dir)


def main():
    # ✅ 策略加载信息
    CONFIG_NAME = "pi0_aloha_sim"
    CACHE_DIR = "/home/admin123/.cache/openpi/openpi-assets/checkpoints/pi0_aloha_sim"
    ACTION_HORIZON = 10
    TASK_NAME = "gym_aloha/AlohaTransferCube-v0"
    SEED = 0

    # ✅ 设置输出目录（项目根目录下，按日期/时间分层）
    now = datetime.now()
    output_dir = pathlib.Path("output/aloha_sim") / now.strftime("%Y-%m-%d") / now.strftime("%H-%M-%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ✅ 加载策略（从 .cache）
    raw_policy = load_policy_from_cache(CONFIG_NAME, CACHE_DIR)
    policy = ActionChunkBroker(raw_policy, action_horizon=ACTION_HORIZON, use_rtc=True)

    # ✅ 初始化仿真环境 + agent
    env = _env.AlohaSimEnvironment(task=TASK_NAME, seed=SEED)
    agent = _policy_agent.PolicyAgent(policy=policy)

    # ✅ 设置 runtime（含视频保存器）
    runner = _runtime.Runtime(
        environment=env,
        agent=agent,
        subscribers=[
            _saver.VideoSaver(output_dir),
        ],
        max_hz=50,
        num_episodes=10,
        use_rtc=True,
    )

    logging.info(f"Starting rollout. Saving results to {output_dir}")
    runner.run()
    logging.info("Rollout completed.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
