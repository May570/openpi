# download_datasets.py
import os, time
from huggingface_hub import snapshot_download
try:
    # 0.33.x 在这里
    from huggingface_hub.utils import HfHubHTTPError
except Exception:
    # 旧版兜底（有些放在 errors）
    try:
        from huggingface_hub.errors import HfHubHTTPError  # type: ignore
    except Exception:
        class HfHubHTTPError(Exception):  # 最后兜底
            pass

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # pip install hf_transfer

for attempt in range(3):
    try:
        snapshot_download(
            repo_id="IPEC-COMMUNITY/bridge_orig_lerobot",
            repo_type="dataset",
            resume_download=True,
            # force_download=True,
            max_workers=10,
        )
        print("download ok")
        break
    except (HfHubHTTPError, OSError) as e:
        if attempt == 2:
            raise
        time.sleep(10)
