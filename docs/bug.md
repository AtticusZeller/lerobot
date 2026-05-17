# Known Issues & Workarounds

## egl-probe 1.0.2 fails to build with CMake 4.x

**Symptom**: `uv sync --extra "libero"` fails at `egl-probe==1.0.2` with:

```
CMake Error at CMakeLists.txt:1 (cmake_minimum_required):
    Compatibility with CMake < 3.5 has been removed from CMake.
```

**Root cause**: `egl-probe` is a transitive dependency (`hf-libero` -> `robomimic` -> `egl-probe`) with an outdated `CMakeLists.txt`. CMake 4.x removed backward compatibility for projects declaring `cmake_minimum_required` below 3.5.

**Workaround**: Set `CMAKE_POLICY_VERSION_MINIMUM=3.5` before running sync:

```bash
CMAKE_POLICY_VERSION_MINIMUM=3.5 uv sync --extra "pi" --extra "test" --extra "libero"
```

To persist across sessions, add to shell profile:

```bash
export CMAKE_POLICY_VERSION_MINIMUM=3.5
```

**Upstream fix needed**: `egl-probe` 1.0.2 (or `robomimic` / `hf-libero`) needs to bump its `cmake_minimum_required`. Tracked packages:
- https://github.com/NVlabs/egl_probe (egl-probe)
- https://github.com/ARISE-Initiative/robomimic (robomimic)

**Last verified**: 2026-05-16, CMake 4.1.3, Python 3.12

---

## LIBERO eval fails: `libEGL.so.0` not found

**Symptom**: `lerobot-eval` (or `dev.sh eval_baseline`) crashes at import with:

```
INFO  ... Failed to load library ( 'libEGL.so.0' ): libEGL.so.0: cannot open shared object file: No such file or directory
...
AttributeError: 'NoneType' object has no attribute 'eglQueryString'
```

Full traceback goes through `robosuite` -> `mujoco.egl` -> `OpenGL.EGL`.

**Root cause**: MuJoCo/robosuite uses EGL for offscreen (headless) rendering. The system-level EGL shared library is not installed. This is a Linux system dependency, not a Python package.

**Fix**:

```bash
apt-get update && apt-get install -y libegl1-mesa libegl1 libgl1-mesa-glx libgles2
```

**When to expect this**: any machine that hasn't previously run MuJoCo with EGL rendering (fresh containers, new cloud instances). GPU drivers alone are not enough — the EGL loader (`libEGL.so.1`) must be present.

**Last verified**: 2026-05-16, Ubuntu 22.04, MuJoCo 3.6.0, robosuite 1.4.0

---

## LIBERO eval fails: `_LazyAsyncVectorEnv` has no attribute `unwrapped`

**Symptom**: `dev.sh eval_baseline` gets past imports and environment startup, then crashes during eval/video bookkeeping with:

```text
AttributeError: '_LazyAsyncVectorEnv' object has no attribute 'unwrapped'
```

Traceback goes through `src/lerobot/scripts/lerobot_eval.py` in `eval_policy()` when reading
`env.unwrapped.metadata["render_fps"]`.

**Root cause**: LIBERO / MetaWorld async eval uses `src/lerobot/envs/utils.py::_LazyAsyncVectorEnv` to defer
`gym.vector.AsyncVectorEnv` creation until first use. The wrapper already proxied `reset()`, `step()`, `call()`,
and `get_attr()`, but it did not proxy the vector-env properties `unwrapped` and `metadata`. `lerobot_eval.py`
expects those properties to exist when compiling episode data and writing videos, so eval crashes even though the
underlying async env is valid.

**Fix**: add `@property` proxies for `metadata` and `unwrapped` on `_LazyAsyncVectorEnv`, both calling `_ensure()`
before forwarding to the real `AsyncVectorEnv`.

**Code path**:

- `src/lerobot/envs/utils.py` — `_LazyAsyncVectorEnv.metadata` and `.unwrapped`
- `src/lerobot/scripts/lerobot_eval.py` — reads `env.unwrapped.metadata["render_fps"]`

**Last verified**: 2026-05-16, with `LIBERO_SUITE=libero_spatial` and `eval_baseline`

---

## `HuggingFaceVLA/libero` episodes metadata has broken `data/file_index`

**Symptom**: any code path that does `LeRobotDataset(repo_id="HuggingFaceVLA/libero", episodes=[...])` (single-task RL Token training, custom filtering) fails with:

```text
ValueError: Instruction "train" corresponds to no data!
```

Even after `snapshot_download` says it fetched the right files. The HF datasets library reports zero rows after the `pa_ds.field("episode_index").isin(episodes)` filter.

**Root cause**: The dataset's `meta/episodes/chunk-000/file-000.parquet` column `data/file_index` maps each episode to a stale chunking layout (69 files of ~24 episodes), but the actual `data/chunk-000/file-XXX.parquet` files on the Hub use the new layout (377 files of ~3–5 episodes). `LeRobotDataset` calls `meta.get_data_file_path(ep)` to pick which parquet to download, gets a wrong filename, downloads that file, and then the `episode_index isin` filter finds nothing because the file actually contains different episodes.

Confirmed upstream: [HuggingFaceVLA/libero community issue #5](https://huggingface.co/datasets/HuggingFaceVLA/libero/discussions/5) ("Wrong data/file_index in episodes metadata", arieli13 Jan 23). Every other field (`episode_index`, `index`, `task_index`, `dataset_from_index/to_index`, `tasks`) is consistent — only `data/file_index` is wrong.

**Fix**: shipped in this fork:

- `experiments/dataset_overrides/HuggingFaceVLA_libero_v3.0_episodes_fix.json` — committed truth table (1693 episodes → correct `(chunk_index, file_index)`)
- `scripts/generate_libero_episode_fix.py` — one-time generator that scans all 377 parquet footers via `HfFileSystem` to rebuild the map (rerun if upstream pushes a new revision)
- `src/lerobot/rltoken/dataset_repair.py` — runtime patcher; reads JSON, rewrites the on-disk `meta/episodes/...parquet` in place (original saved to `*.broken.bak`); idempotent on reruns
- hook in `src/lerobot/rltoken/train_token.py:_resolve_episode_filter` calls `repair_episodes_metadata(repo_id, meta.root)` before any `LeRobotDataset(episodes=[...])` is instantiated

To regenerate the JSON if upstream changes:

```bash
uv run python scripts/generate_libero_episode_fix.py \
  --repo-id HuggingFaceVLA/libero --revision v3.0 \
  --expected-total-episodes 1693 \
  --output experiments/dataset_overrides/HuggingFaceVLA_libero_v3.0_episodes_fix.json
```

**Spot-check**:

| episode_index | broken metadata says file_index | actual file (and what our JSON has) |
|---|---|---|
| 137 | 9 | 55 |
| 1272 | 55 | 309 |
| 1692 | 68 | 376 |

**Last verified**: 2026-05-17, revision `v3.0` (commit `86958911c0f959db2bbbdb107eb3e17c5f9c798e`).

---

## RL Token prefix-only π0.5 embedding fails with SDPA dtype mismatch

**Symptom**: `dev.sh train_token --dataset.task_index=0 --steps=20` reaches the first batch, then crashes inside Gemma attention:

```text
RuntimeError: invalid dtype for bias - should match query's dtype
```

Traceback goes through `src/lerobot/rltoken/rl_token.py:extract_vlm_embeddings` -> `PiGemmaModel.forward` -> `transformers/integrations/sdpa_attention.py`.

**Root cause**: RL Token uses a prefix-only π0.5 forward to extract visual hidden states. The prefix embeddings start as fp32, but the π0.5 language model weights are bf16; `PiGemmaModel.forward` casts hidden states to bf16 while the attention bias/mask remains fp32. Transformers' SDPA path requires the bias dtype to match query dtype.

**Fix**: follow `~/DevSpace/rlt-openpi/src/rlt_openpi/vla/embedding_extractor.py` at the RL Token adapter boundary:

- set `language_model.config._attn_implementation = "eager"` for the prefix-only pass
- cast `prefix_embs` to the language model `q_proj.weight.dtype`
- cast `attn_4d` to the same dtype

This is intentionally scoped to `src/lerobot/rltoken/rl_token.py:extract_vlm_embeddings`; do not patch `src/lerobot/policies/pi05/` or `src/lerobot/policies/pi_gemma.py` unless the upstream policy path itself needs a broader fix.

**Last verified**: 2026-05-17, `libero_spatial task 0` Stage 1 smoke path.
