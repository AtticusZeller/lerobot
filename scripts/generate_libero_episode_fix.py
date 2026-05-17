"""Generate the episode→file fix JSON for HuggingFaceVLA/libero.

Upstream `meta/episodes/*.parquet` ships a broken `data/file_index` column
(community issue #5). This script reads each data parquet's footer
(`episode_index` row-group statistics) via `HfFileSystem`, builds the real
`episode_index → (chunk_index, file_index)` map, and writes a JSON sidecar
that `lerobot.rltoken.dataset_repair` consumes at runtime.

Usage:
    uv run python scripts/generate_libero_episode_fix.py \
        --repo-id HuggingFaceVLA/libero --revision v3.0 \
        --output experiments/dataset_overrides/HuggingFaceVLA_libero_v3.0_episodes_fix.json
"""

import argparse
import json
import re
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

CHUNK_RE = re.compile(r"chunk-(\d{3})")
FILE_RE = re.compile(r"file-(\d{3})\.parquet$")


def parse_chunk_file(path: str) -> tuple[int, int]:
    chunk_match = CHUNK_RE.search(path)
    file_match = FILE_RE.search(path)
    if chunk_match is None or file_match is None:
        raise ValueError(f"Unrecognized parquet path: {path}")
    return int(chunk_match.group(1)), int(file_match.group(1))


def list_data_parquets(repo_id: str, revision: str) -> list[str]:
    api = HfApi()
    files = api.list_repo_files(repo_id, repo_type="dataset", revision=revision)
    paths = sorted(f for f in files if f.startswith("data/") and f.endswith(".parquet"))
    if not paths:
        raise RuntimeError(f"No data parquet files found in {repo_id}@{revision}")
    return paths


def episode_min_max(metadata: pq.FileMetaData) -> tuple[int, int]:
    col_idx = metadata.schema.names.index("episode_index")
    mins, maxs = [], []
    for rg in range(metadata.num_row_groups):
        stats = metadata.row_group(rg).column(col_idx).statistics
        if stats is None or stats.min is None or stats.max is None:
            raise RuntimeError("Parquet row group missing episode_index statistics")
        mins.append(int(stats.min))
        maxs.append(int(stats.max))
    return min(mins), max(maxs)


def build_episode_map(repo_id: str, revision: str) -> dict[int, tuple[int, int]]:
    fs = HfFileSystem()
    paths = list_data_parquets(repo_id, revision)
    ep_to_file: dict[int, tuple[int, int]] = {}
    for i, path in enumerate(paths, 1):
        chunk_idx, file_idx = parse_chunk_file(path)
        with fs.open(f"datasets/{repo_id}@{revision}/{path}", "rb") as fh:
            md = pq.read_metadata(fh)
        ep_min, ep_max = episode_min_max(md)
        for ep in range(ep_min, ep_max + 1):
            if ep in ep_to_file and ep_to_file[ep] != (chunk_idx, file_idx):
                raise RuntimeError(
                    f"episode_index {ep} appears in two files: "
                    f"{ep_to_file[ep]} and ({chunk_idx}, {file_idx})"
                )
            ep_to_file[ep] = (chunk_idx, file_idx)
        if i % 25 == 0 or i == len(paths):
            print(f"  scanned {i}/{len(paths)} parquet footers ({path}: eps {ep_min}-{ep_max})")
    return ep_to_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", default="v3.0")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--expected-total-episodes",
        type=int,
        default=None,
        help="If set, sanity-check that the map covers exactly this many episodes.",
    )
    args = parser.parse_args()

    print(f"Scanning {args.repo_id}@{args.revision} ...")
    ep_to_file = build_episode_map(args.repo_id, args.revision)
    sorted_eps = sorted(ep_to_file)
    print(f"Total episodes mapped: {len(sorted_eps)} (range {sorted_eps[0]}..{sorted_eps[-1]})")

    expected = sorted_eps[-1] - sorted_eps[0] + 1
    if len(sorted_eps) != expected:
        missing = sorted(set(range(sorted_eps[0], sorted_eps[-1] + 1)) - set(sorted_eps))
        raise RuntimeError(f"Gap in episode coverage; missing {len(missing)} episodes, first 10: {missing[:10]}")
    if args.expected_total_episodes is not None and len(sorted_eps) != args.expected_total_episodes:
        raise RuntimeError(
            f"Expected {args.expected_total_episodes} episodes, got {len(sorted_eps)}."
        )

    payload = {
        "repo_id": args.repo_id,
        "revision": args.revision,
        "fix_column": "data/file_index",
        "entries": [
            {
                "episode_index": ep,
                "data/chunk_index": ep_to_file[ep][0],
                "data/file_index": ep_to_file[ep][1],
            }
            for ep in sorted_eps
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {args.output} ({args.output.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
