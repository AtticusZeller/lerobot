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
