# robust_download_bridge_dataset.py
import time
from huggingface_hub import snapshot_download
try:
    from huggingface_hub.utils import HfHubHTTPError
except ImportError:
    from huggingface_hub.errors import HfHubHTTPError

# ========================= 配置 =========================
REPO_ID = "google/paligemma-3b-pt-224"
LOCAL_DIR = "google/paligemma-3b-pt-224"   # 导出目录
MAX_RETRIES = 100                           # 最大尝试次数
SLEEP_SECONDS = 10                        # 每次失败后等待秒数
MAX_WORKERS = 8                            # 并发线程数，4~8比较稳
# ========================================================

for i in range(1, MAX_RETRIES + 1):
    try:
        print(f"\n=== 尝试 {i}/{MAX_RETRIES} ===")
        path = snapshot_download(
            repo_id=REPO_ID,
            repo_type="model",
            local_dir=LOCAL_DIR,
            max_workers=MAX_WORKERS,
        )
        print(f"✅ 下载完成：{path}")
        break
    except (HfHubHTTPError, OSError, Exception) as e:
        print(f"⚠️ 第 {i} 次失败：{e}")
        if i < MAX_RETRIES:
            print(f"等待 {SLEEP_SECONDS} 秒后重试...")
            time.sleep(SLEEP_SECONDS)
        else:
            print("❌ 达到最大尝试次数，仍未完成。")
