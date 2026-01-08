from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, HF_LEROBOT_HOME
from pathlib import Path
import shutil

import os
os.environ["HF_LEROBOT_HOME"] = "/share/project/wujiling/openpi/datasets/"


def load_dataset_with_root(repo_path: Path):
    repo_path = repo_path.resolve()
    repo_id = repo_path.name

    print(f"Loading dataset: repo_id={repo_id}, root={repo_path}")
    return LeRobotDataset(
        repo_id=repo_id,
        root=repo_path,
        download_videos=False,
    )


def merge_repos(output_repo_id: str, input_paths: list[str], output_root: str | None = None):
    # 处理输入路径
    input_paths = [Path(p).resolve() for p in input_paths]

    # 决定输出 root
    if output_root is None:
        out_root = HF_LEROBOT_HOME
    else:
        out_root = Path(output_root).resolve()

    out_dir = out_root / output_repo_id

    # 如果已存在则删除
    if out_dir.exists():
        print("Deleting existing dataset:", out_dir)
        shutil.rmtree(out_dir)

    # 加载第一个数据集作为 schema 参考
    base_ds = load_dataset_with_root(input_paths[0])

    print("Creating merged repo:", output_repo_id)
    merged = LeRobotDataset.create(
        repo_id=output_repo_id,
        root=out_root,
        fps=base_ds.fps,
        robot_type="A2D",
        features=base_ds.features,
        use_videos=False,
        video_backend=base_ds.video_backend,
    )

    # 合并其余数据
    for p in input_paths:
        print("Merging:", p)
        ds = load_dataset_with_root(p)

        hf = ds.hf_dataset
        current_ep = None

        for row in hf:
            ep_idx = int(row["episode_index"])

            # 如果是新 episode，则先保存上一个 episode
            if current_ep is None:
                current_ep = ep_idx
            elif ep_idx != current_ep:
                merged.save_episode()
                current_ep = ep_idx

            # 添加这一帧（row 是 dict）
            merged.add_frame(row)

        # 最后一个 episode 别忘记保存
        if current_ep is not None:
            merged.save_episode()

    print("Merge complete!")
    print("Output saved to:", out_dir)


if __name__ == "__main__":
    merge_repos(
        output_repo_id="pour_coffee_merged",
        input_paths=[
            "/share/project/wujiling/data/pour/pour_coffee_3924",
            "/share/project/wujiling/data/pour/pour_coffee_3926",
            "/share/project/wujiling/data/pour/pour_coffee_3928",
        ],
        # 这里可以指定最终输出位置，如果不填，则默认写到 $HF_LEROBOT_HOME
        output_root="/share/project/wujiling/raw_data/lerobot_datasets",
    )