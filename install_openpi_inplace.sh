#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="simpler_env"          # 你的现有环境名
CUDA_FLAVOR="${1:-cpu}"    # 可选: cpu 或 cu124（CUDA 12.4 版 PyTorch）

# 让 conda activate 生效
if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda 不在 PATH"; exit 1
fi
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

conda activate "${ENV_NAME}"

echo "[1/6] 升级 Python 到 3.11（openpi 需要 >=3.11）"
conda install -y python=3.11  # pyproject: requires-python >=3.11
python -m pip install -U pip wheel setuptools

echo "[2/6] 卸载潜在冲突（旧 torch/jax/flax/transformers）"
python - <<'PY'
import subprocess, sys
pkgs = ["jax","jaxlib","flax","torch","torchvision","torchaudio","transformers"]
subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", *pkgs])
PY

echo "[3/6] 安装 PyTorch 2.7.1（按需 CPU 或 CUDA12.4）"
if [[ "${CUDA_FLAVOR}" == "cu124" ]]; then
  python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.7.1 torchvision torchaudio
else
  python -m pip install torch==2.7.1 torchvision torchaudio
fi
# ↑ openpi: torch==2.7.1
#   https://pytorch.org/get-started/locally/ 选择对应 CUDA 轮子

echo "[4/6] 安装 openpi 其余核心依赖（严格按 pyproject.toml）"
python -m pip install \
  "augmax>=0.3.4" \
  "dm-tree>=0.1.8" \
  "einops>=0.8.0" \
  "equinox>=0.11.8" \
  "flatbuffers>=24.3.25" \
  "flax==0.10.2" \
  "fsspec[gcs]>=2024.6.0" \
  "gym-aloha>=0.1.1" \
  "imageio>=2.36.1" \
  "jaxtyping==0.2.36" \
  "ml_collections==1.0.0" \
  "numpy>=1.22.4,<2.0.0" \
  "numpydantic>=1.6.6" \
  "opencv-python>=4.10.0.84" \
  "orbax-checkpoint==0.11.13" \
  "pillow>=11.0.0" \
  "sentencepiece>=0.2.0" \
  "tqdm-loggable>=0.2" \
  "typing-extensions>=4.12.2" \
  "tyro>=0.9.5" \
  "wandb>=0.19.1" \
  "filelock>=3.16.1" \
  "beartype==0.19.0" \
  "treescope>=0.1.7" \
  "transformers==4.53.2" \
  "rich>=14.0.0" \
  "polars>=1.30.0"
# 以上逐条对应 pyproject [project.dependencies] 列表
# 参考：pyproject.toml L8-L40

echo "[5/6] 安装 JAX（CUDA12 版，驱动不够可改 jax[cpu]==0.5.3）"
python -m pip install "jax[cuda12]==0.5.3"

echo "[5.1/6] 对齐 uv 的 override（强钉版本以免 ABI/行为漂移）"
python -m pip install "ml-dtypes==0.4.1" "tensorstore==0.1.74"
# 参考：pyproject [tool.uv.override-dependencies]

echo "[5.2/6] 安装 lerobot（指定 git 修订）"
python -m pip install "git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"
# 参考：pyproject [tool.uv.sources].lerobot rev 同步

echo "[6/6] 可选：需要 RLDS/TF 时再装（对应 dependency-group rlds）"
echo "    python -m pip install tensorflow-cpu==2.15.0 tensorflow-datasets==4.9.9 \\"
echo "      git+https://github.com/kvablack/dlimp@ad72ce3a9b414db2185bc0b38461d4101a65477a"

echo "[自检]"
python - <<'PY'
import importlib, jax, torch, sys
print("Python:", sys.version.split()[0])
for m in ["jax","flax","torch","transformers","numpy","orbax.checkpoint","polars"]:
    try: importlib.import_module(m); print("[OK]", m)
    except Exception as e: print("[FAIL]", m, "->", e)
print("JAX devices:", jax.devices())
print("Torch CUDA available?", torch.cuda.is_available())
PY

echo "[完成] 当前环境已就地集成 openpi 依赖。"

