"""Rebuild a LeRobot dataset with head/tail idle frames trimmed per episode.

Reads a ``keep_ranges.json`` produced by ``filter_idle_frames.py`` (single
``[start, end)`` range per episode), then iterates the source dataset and
materializes a new trimmed dataset. Encoding goes through ``vcodec="auto"`` by
default, which picks the fastest available hardware encoder (NVENC on NVIDIA
GPUs, VideoToolbox on macOS, VA-API / QSV on Intel/AMD) and falls back to
``libsvtav1`` only if none is present.

Episodes listed in ``--skip-episodes`` (or in the report's ``to_delete``) are
excluded entirely. All other episodes are copied frame-by-frame within their
keep range.

Example:

    python -m lerobot.data_processing.rebuild_trimmed_dataset \
        --src-repo-id Atticuxz/so101-table-cleanup \
        --dst-repo-id Atticuxz/so101-table-cleanup-clean \
        --keep-ranges outputs/idle_filter/keep_ranges.json \
        --skip-episodes-from outputs/idle_filter/report.json \
        --vcodec h264_nvenc \
        --push-to-hub
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def _to_hwc_if_image(key: str, value, features: dict):
    """add_frame expects HWC uint8; LeRobotDataset.__getitem__ yields CHW float tensors."""
    feat = features.get(key)
    if feat is None or feat.get("dtype") not in {"image", "video"}:
        return value
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray) and value.ndim == 3 and value.shape[0] == 3:
        value = np.transpose(value, (1, 2, 0))
    if isinstance(value, np.ndarray) and value.dtype != np.uint8 and value.max() <= 1.0:
        value = (value * 255).astype(np.uint8)
    return value


def load_skip_set(path: Path | None, extra: list[int]) -> set[int]:
    skip: set[int] = set(extra)
    if path is not None:
        data = json.loads(path.read_text())
        skip.update(data.get("to_delete", []))
    return skip


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src-repo-id", required=True)
    p.add_argument("--src-root", default=None)
    p.add_argument("--dst-repo-id", required=True, help="New repo ID for the trimmed dataset")
    p.add_argument("--dst-root", default=None, help="Local output root (defaults to HF cache)")
    p.add_argument("--keep-ranges", required=True, help="Path to keep_ranges.json from filter_idle_frames")
    p.add_argument(
        "--skip-episodes-from",
        default=None,
        help="Path to a report.json containing a 'to_delete' list (idle filter or integrity report)",
    )
    p.add_argument(
        "--skip-episodes",
        type=int,
        nargs="*",
        default=[],
        help="Extra episode indices to drop on top of the report",
    )
    p.add_argument(
        "--vcodec",
        default="auto",
        help="Video codec: auto | h264_nvenc | hevc_nvenc | libsvtav1 | h264 | hevc | ...",
    )
    p.add_argument("--push-to-hub", action="store_true")
    args = p.parse_args()

    src = LeRobotDataset(args.src_repo_id, root=args.src_root)
    keep_ranges: dict[str, list[int]] = json.loads(Path(args.keep_ranges).read_text())
    skip = load_skip_set(
        Path(args.skip_episodes_from) if args.skip_episodes_from else None,
        args.skip_episodes,
    )

    print(f"Source: {args.src_repo_id}  episodes={src.meta.total_episodes}  fps={src.meta.fps}")
    print(f"Skipping {len(skip)} episode(s): {sorted(skip)}")
    print(f"Using vcodec: {args.vcodec}")

    dst = LeRobotDataset.create(
        repo_id=args.dst_repo_id,
        fps=src.meta.fps,
        root=args.dst_root,
        features=src.features,
        robot_type=src.meta.robot_type,
        use_videos=True,
        vcodec=args.vcodec,
    )

    ep_indices = np.asarray(src.hf_dataset["episode_index"])

    for ep_idx in tqdm(range(src.meta.total_episodes), desc="Episodes"):
        if ep_idx in skip:
            continue
        rng = keep_ranges.get(str(ep_idx))
        if rng is None:
            continue
        start, end = int(rng[0]), int(rng[1])
        if end <= start:
            continue

        global_idx = np.where(ep_indices == ep_idx)[0]
        if len(global_idx) == 0:
            continue
        selected = global_idx[start:end]

        first_task_idx = int(src[int(global_idx[0])]["task_index"])
        task = src.meta.tasks.iloc[first_task_idx].name

        for gi in selected:
            frame = src[int(gi)]
            payload = {
                k: _to_hwc_if_image(k, v, src.features)
                for k, v in frame.items()
                if k not in {
                    "index",
                    "frame_index",
                    "episode_index",
                    "timestamp",
                    "task_index",
                    "task",
                    "subtask_index",
                }
            }
            payload["task"] = task
            dst.add_frame(payload)
        dst.save_episode()

    dst.finalize()
    print(f"\nTrimmed dataset saved to: {dst.root}")
    print(f"Episodes written: {dst.meta.total_episodes}, total frames: {dst.meta.total_frames}")

    if args.push_to_hub:
        dst.push_to_hub()
        print(f"Pushed to hub: {args.dst_repo_id}")


if __name__ == "__main__":
    main()
