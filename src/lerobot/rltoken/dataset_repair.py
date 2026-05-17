"""Patch broken `data/file_index` in upstream LeRobot dataset metadata.

Some upstream LeRobot datasets (notably ``HuggingFaceVLA/libero`` v3.0,
community issue #5) ship a `meta/episodes/*.parquet` whose `data/file_index`
column does not match the actual data parquet layout. This breaks
`LeRobotDataset(episodes=[...])` because `_download` fetches the wrong files
and the `episode_index isin` filter then returns zero rows.

We keep a committed JSON sidecar with the correct mapping (built offline by
``scripts/generate_libero_episode_fix.py``) and rewrite the on-disk
episodes parquet in place before any downstream consumer reads it. The
original is renamed to ``*.broken.bak`` on first patch; subsequent runs are
no-ops if the parquet already matches the sidecar.
"""

import json
import logging
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.datasets.utils import EPISODES_DIR

_REPO_ROOT = Path(__file__).resolve().parents[3]
_OVERRIDES_DIR = _REPO_ROOT / "experiments" / "dataset_overrides"

OVERRIDE_REGISTRY: dict[str, Path] = {
    "HuggingFaceVLA/libero": _OVERRIDES_DIR / "HuggingFaceVLA_libero_v3.0_episodes_fix.json",
}


def _load_override(repo_id: str) -> dict | None:
    path = OVERRIDE_REGISTRY.get(repo_id)
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text())


def _find_episodes_parquets(root: Path) -> list[Path]:
    return sorted((root / EPISODES_DIR).glob("*/*.parquet"))


def _apply_fix_to_parquet(parquet_path: Path, ep_to_chunk_file: dict[int, tuple[int, int]]) -> bool:
    """Rewrite `data/chunk_index` and `data/file_index` for episodes in this parquet.

    Returns True if the file was modified, False if it already matched.
    """
    table = pq.read_table(parquet_path)
    episode_col = table.column("episode_index").to_pylist()

    current_chunk = table.column("data/chunk_index").to_pylist()
    current_file = table.column("data/file_index").to_pylist()
    new_chunk: list[int] = []
    new_file: list[int] = []
    changed = False
    for ep, ck, fk in zip(episode_col, current_chunk, current_file, strict=True):
        target = ep_to_chunk_file.get(int(ep))
        if target is None:
            # Episode not in override map: keep existing value but log loudly later.
            new_chunk.append(ck)
            new_file.append(fk)
            continue
        new_chunk.append(target[0])
        new_file.append(target[1])
        if target != (ck, fk):
            changed = True

    if not changed:
        return False

    new_table = table.set_column(
        table.schema.get_field_index("data/chunk_index"),
        "data/chunk_index",
        pa.array(new_chunk, type=table.schema.field("data/chunk_index").type),
    )
    new_table = new_table.set_column(
        new_table.schema.get_field_index("data/file_index"),
        "data/file_index",
        pa.array(new_file, type=table.schema.field("data/file_index").type),
    )

    backup = parquet_path.with_suffix(parquet_path.suffix + ".broken.bak")
    if not backup.exists():
        # Atomic on POSIX; preserves original blob symlink target intact at the bak path.
        os.rename(parquet_path, backup)
    else:
        # Backup already exists from a prior run; the current parquet is our patched copy
        # being re-patched (e.g. registry update). Just remove it before writing the new one.
        parquet_path.unlink()

    tmp_path = parquet_path.with_suffix(parquet_path.suffix + ".tmp")
    pq.write_table(new_table, tmp_path)
    os.replace(tmp_path, parquet_path)
    return True


def repair_episodes_metadata(repo_id: str, root: Path) -> bool:
    """Apply the registered episode-mapping fix for ``repo_id`` if one exists.

    Args:
        repo_id: Dataset repo identifier (e.g. ``"HuggingFaceVLA/libero"``).
        root: Local dataset root containing the ``meta/`` directory.

    Returns:
        ``True`` if any parquet was rewritten in this call, ``False`` if no
        override is registered for this repo or the metadata already matched.
    """
    override = _load_override(repo_id)
    if override is None:
        return False
    if override.get("fix_column") != "data/file_index":
        logging.warning("Unknown fix_column in override for %s: %s", repo_id, override.get("fix_column"))
        return False

    ep_to_chunk_file: dict[int, tuple[int, int]] = {
        int(entry["episode_index"]): (int(entry["data/chunk_index"]), int(entry["data/file_index"]))
        for entry in override["entries"]
    }

    parquets = _find_episodes_parquets(root)
    if not parquets:
        logging.warning("No episodes parquet found under %s; skipping repair.", root / EPISODES_DIR)
        return False

    any_changed = False
    for parquet_path in parquets:
        if _apply_fix_to_parquet(parquet_path, ep_to_chunk_file):
            logging.info("Patched broken data/file_index in %s", parquet_path)
            any_changed = True
    return any_changed
