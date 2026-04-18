"""Check a LeRobot dataset for metadata / video inconsistencies.

Currently covers one failure mode that occurs with streaming-encoded datasets:
the per-episode ``length`` in ``meta/episodes/*.parquet`` disagrees with the
actual video frame count derived from
``(to_timestamp - from_timestamp) * fps``. When this happens,
``lerobot-edit-dataset delete_episodes`` will abort with
``AssertionError: Episode length mismatch``, and the frames past the video's
end cannot be decoded during training.

Outputs:

* ``integrity_report.json`` — per-episode mismatches and a ``to_delete`` list.

Example:

    python -m lerobot.data_processing.check_dataset_integrity \
        --repo-id ${HF_USER}/so101-table-cleanup \
        --output-dir outputs/integrity
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-id", required=True)
    p.add_argument("--root", default=None)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = LeRobotDataset(args.repo_id, root=args.root)
    fps = dataset.meta.fps
    num_episodes = dataset.meta.total_episodes

    mismatches: list[dict] = []
    bad_episodes: set[int] = set()

    for ep_idx in range(num_episodes):
        ep = dataset.meta.episodes[ep_idx]
        parquet_len = int(ep["length"])
        for key in list(ep.keys()):
            if not (key.startswith("videos/") and key.endswith("/from_timestamp")):
                continue
            video_key = key.split("/")[1]
            from_ts = ep[f"videos/{video_key}/from_timestamp"]
            to_ts = ep[f"videos/{video_key}/to_timestamp"]
            video_frames = round((to_ts - from_ts) * fps)
            if parquet_len != video_frames:
                mismatches.append(
                    {
                        "episode_index": ep_idx,
                        "camera": video_key,
                        "parquet_len": parquet_len,
                        "video_frames": video_frames,
                        "diff": parquet_len - video_frames,
                    }
                )
                bad_episodes.add(ep_idx)

    to_delete = sorted(bad_episodes)
    report = {
        "repo_id": args.repo_id,
        "num_episodes": num_episodes,
        "fps": fps,
        "num_mismatches": len(mismatches),
        "to_delete": to_delete,
        "mismatches": mismatches,
    }
    (out_dir / "integrity_report.json").write_text(json.dumps(report, indent=2))

    print(f"Scanned {num_episodes} episodes.")
    print(f"Report: {out_dir / 'integrity_report.json'}")
    if not mismatches:
        print("No inconsistencies found.")
        return

    print(f"\n{len(mismatches)} mismatches across {len(to_delete)} episodes:")
    print(f"{'ep':>4} {'camera':<30} {'parquet':>8} {'video':>8} {'diff':>6}")
    for m in mismatches:
        print(
            f"{m['episode_index']:>4} {m['camera']:<30} "
            f"{m['parquet_len']:>8} {m['video_frames']:>8} {m['diff']:>+6}"
        )

    print(f"\nEpisodes to delete: {to_delete}")
    print("\nTo remove them:")
    print(
        f"  lerobot-edit-dataset \\\n"
        f"    --repo_id {args.repo_id} \\\n"
        f"    --operation.type delete_episodes \\\n"
        f'    --operation.episode_indices "{to_delete}"'
    )


if __name__ == "__main__":
    main()
