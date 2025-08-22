# conda activate /share/project/lvhuaihai/envs/openpi
export WANDB_MODE=offline
export HF_LEROBOT_HOME=/share/project/section
export HF_DATASETS_CACHE=/share/project/lvhuaihai/lerobot_data/cache
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
python scripts/train.py pi0_agilex_orange_norm --exp-name=pi0_agilex_orange_norm --batch_size=8 --fsdp_devices=8 --overwrite