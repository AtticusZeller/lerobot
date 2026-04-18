"""Detect leading / trailing idle segments in a LeRobot dataset and flag
episodes that are mostly idle for deletion.

Design choices (differs from openpi's DROID filter):

* **Only head / tail idle is trimmed.** Middle idle runs are kept as-is because
  splitting a trajectory mid-episode breaks temporal continuity for
  action-chunking policies.
* **Whole-episode deletion.** If the non-idle ratio of an episode (after head
  and tail trimming) falls below ``--min-keep-ratio``, the whole episode is
  marked for deletion — because an episode with too much internal stalling
  usually indicates a botched demo and should be re-recorded.

Outputs (written to ``--output-dir``):

1. ``keep_ranges.json`` — ``{episode_index: [start, end]}`` single half-open
   range per episode (head/tail trimmed).
2. ``report.json`` — per-episode stats + list of episodes to delete.

Example:

    python -m lerobot.data_processing.filter_idle_frames \
        --repo-id ${HF_USER}/so101-table-cleanup \
        --output-dir outputs/idle_filter \
        --signal observation.state \
        --idle-threshold 1e-3 \
        --min-keep-ratio 0.3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def trim_head_tail_idle(
    signal: np.ndarray,
    idle_threshold: float,
    joint_range: np.ndarray | None = None,
    smoothing_window: int = 1,
) -> tuple[int, int]:
    """Return (start, end) half-open range with leading/trailing idle trimmed.

    Per-joint delta is normalized by ``joint_range`` (max - min observed across
    the full dataset) so a single dimensionless threshold works across joints
    with different physical ranges. A frame is "moving" when the max normalized
    delta across all joints >= ``idle_threshold``.

    If ``smoothing_window > 1``, the normalized delta is averaged over a rolling
    window, which smooths out single-frame servo jitter.
    """
    if len(signal) < 2:
        return (0, len(signal))

    delta = np.abs(np.diff(signal, axis=0))  # (T-1, N_joints)
    if joint_range is not None:
        safe_range = np.where(joint_range > 1e-6, joint_range, 1.0)
        delta = delta / safe_range  # normalize per-joint

    per_frame = np.max(delta, axis=1)  # (T-1,)

    if smoothing_window > 1:
        kernel = np.ones(smoothing_window) / smoothing_window
        per_frame = np.convolve(per_frame, kernel, mode="same")

    moving = per_frame >= idle_threshold

    if not moving.any():
        return (0, 0)

    first = int(np.argmax(moving))
    last = int(len(moving) - 1 - np.argmax(moving[::-1]))

    start = first
    end = last + 2
    return (start, min(end, len(signal)))


def _parquet_path(root: Path, episode_index: int, chunks_size: int) -> Path:
    chunk = episode_index // chunks_size
    return root / f"data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-id", required=True)
    p.add_argument("--root", default=None, help="Local dataset root (optional)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--signal", default="observation.state", help="Feature key to diff on")
    p.add_argument(
        "--idle-threshold",
        type=float,
        default=0.005,
        help=(
            "Dimensionless threshold on per-joint normalized delta. "
            "0.005 = any joint moved ≥0.5%% of its full range between frames."
        ),
    )
    p.add_argument(
        "--smoothing-window",
        type=int,
        default=3,
        help="Rolling-mean window size over the normalized delta (suppresses single-frame jitter).",
    )
    p.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable per-joint normalization (use raw delta magnitude in original units).",
    )
    p.add_argument(
        "--min-keep-ratio",
        type=float,
        default=0.3,
        help=(
            "If (kept_frames / total_frames) < this after head/tail trim, mark the episode for deletion."
            " Default 0.3 means episodes with >70%% idle are flagged."
        ),
    )
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use LeRobotDataset only for metadata — never touch hf_dataset in the loop
    dataset = LeRobotDataset(args.repo_id, root=args.root)
    num_episodes = dataset.meta.total_episodes
    chunks_size = dataset.meta.info.get("chunks_size", 1000)
    root = Path(dataset.root)

    # Per-joint range from dataset stats, used to normalize idle threshold across joints
    joint_range: np.ndarray | None = None
    if not args.no_normalize:
        all_stats = dataset.meta.stats or {}
        stats = all_stats.get(args.signal)
        if stats is None or "max" not in stats or "min" not in stats:
            print(f"WARN: no stats for '{args.signal}', falling back to raw delta (per-joint unnormalized).")
        else:
            joint_range = np.asarray(stats["max"]) - np.asarray(stats["min"])
            print(f"Per-joint range for '{args.signal}': {np.round(joint_range, 2).tolist()}")

    keep_ranges: dict[str, list[int]] = {}
    report: dict[str, dict] = {}
    to_delete: list[int] = []

    for ep_idx in tqdm(range(num_episodes), desc="Scanning episodes"):
        pq_path = _parquet_path(root, ep_idx, chunks_size)
        df = pd.read_parquet(pq_path, columns=[args.signal])
        signal = np.stack(df[args.signal].tolist())

        total = len(signal)
        start, end = trim_head_tail_idle(
            signal,
            idle_threshold=args.idle_threshold,
            joint_range=joint_range,
            smoothing_window=args.smoothing_window,
        )
        kept = end - start
        keep_ratio = kept / total if total > 0 else 0.0

        keep_ranges[str(ep_idx)] = [start, end]
        report[str(ep_idx)] = {
            "total_frames": total,
            "kept_frames": kept,
            "head_trimmed": start,
            "tail_trimmed": total - end,
            "keep_ratio": round(keep_ratio, 3),
        }
        if keep_ratio < args.min_keep_ratio or kept == 0:
            to_delete.append(ep_idx)

    (out_dir / "keep_ranges.json").write_text(json.dumps(keep_ranges, indent=2))
    (out_dir / "report.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "num_episodes": num_episodes,
                "to_delete": to_delete,
                "per_episode": report,
            },
            indent=2,
        )
    )

    total_head = sum(r["head_trimmed"] for r in report.values())
    total_tail = sum(r["tail_trimmed"] for r in report.values())
    print(f"\nScanned {num_episodes} episodes.")
    print(f"Total leading idle trimmed:  {total_head} frames")
    print(f"Total trailing idle trimmed: {total_tail} frames")
    print(f"Kept ranges: {out_dir / 'keep_ranges.json'}")
    print(f"Report:      {out_dir / 'report.json'}")
    if to_delete:
        print(
            f"\n{len(to_delete)} episodes have keep_ratio < {args.min_keep_ratio} "
            f"(too much internal idle — recommend re-recording):"
        )
        print(f"  {to_delete}")
        print("\nTo remove them:")
        print(
            f"  lerobot-edit-dataset \\\n"
            f"    --repo_id {args.repo_id} \\\n"
            f"    --operation.type delete_episodes \\\n"
            f'    --operation.episode_indices "{to_delete}"'
        )


if __name__ == "__main__":
    main()
