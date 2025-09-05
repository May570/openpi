from openpi.shared import download

checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi0_base")
print("Checkpoint is here:", checkpoint_dir)