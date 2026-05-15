"""阶段一基线：跑冻结 π0.5 在 LIBERO 四个子集上的吞吐率 + 步数分布。

设计 / 指标定义：见 ``docs/rltoken_plan.md`` §三。
本入口是 ``lerobot.scripts.lerobot_eval`` 的薄壳——复用同一个 policy / preprocessor / postprocessor，
只是在多个 suite 上各跑一遍并按 suite 汇总结果。
"""

from __future__ import annotations

import csv
import json
import logging
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pformat

import draccus
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import LiberoEnv
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import close_envs
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import init_logging

DEFAULT_SUITES: tuple[str, ...] = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
)


@dataclass
class ThroughputEvalConfig:
    policy_path: str = "lerobot/pi05_libero"
    suites: list[str] = field(default_factory=lambda: list(DEFAULT_SUITES))
    n_episodes: int = 50
    batch_size: int = 5
    use_async_envs: bool = False
    max_parallel_tasks: int = 1
    max_episodes_rendered: int = 0
    output_dir: str = "outputs/baseline"
    device: str | None = None
    use_amp: bool = False
    seed: int = 1000
    trust_remote_code: bool = False


def _suite_summary(group_metrics: dict, n_episodes: int) -> dict:
    return {
        "n_episodes": group_metrics.get("n_episodes", n_episodes),
        "pc_success": group_metrics.get("pc_success", float("nan")),
        "avg_episode_length": group_metrics.get("avg_episode_length", float("nan")),
        "avg_success_length": group_metrics.get("avg_success_length", float("nan")),
        "p25_episode_length": group_metrics.get("p25_episode_length", float("nan")),
        "p50_episode_length": group_metrics.get("p50_episode_length", float("nan")),
        "p75_episode_length": group_metrics.get("p75_episode_length", float("nan")),
        "throughput": group_metrics.get("throughput", float("nan")),
        "total_env_steps": group_metrics.get("total_env_steps", 0),
    }


def _dump_episode_steps_csv(per_task_infos: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task_id", "episode_ix", "success", "episode_length", "sum_reward"])
        for task_info in per_task_infos:
            tid = task_info["task_id"]
            metrics = task_info["metrics"]
            lengths = metrics.get("episode_lengths") or []
            successes = metrics.get("successes") or []
            rewards = metrics.get("sum_rewards") or []
            for ep_ix, (success, length, reward) in enumerate(zip(successes, lengths, rewards, strict=False)):
                writer.writerow([tid, ep_ix, int(bool(success)), int(length), float(reward)])


def run_one_suite(
    suite: str,
    cfg: ThroughputEvalConfig,
    policy,
    preprocessor,
    postprocessor,
    device: torch.device,
) -> dict:
    env_cfg = LiberoEnv(task=suite)
    logging.info(f"[{suite}] building envs (batch_size={cfg.batch_size}, async={cfg.use_async_envs})")
    envs = make_env(
        env_cfg,
        n_envs=cfg.batch_size,
        use_async_envs=cfg.use_async_envs,
        trust_remote_code=cfg.trust_remote_code,
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg, policy_cfg=policy.config
    )

    videos_dir: Path | None = None
    if cfg.max_episodes_rendered > 0:
        videos_dir = Path(cfg.output_dir) / suite / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)

    autocast_ctx = torch.autocast(device_type=device.type) if cfg.use_amp else nullcontext()
    start = time.time()
    with torch.no_grad(), autocast_ctx:
        info = eval_policy_all(
            envs=envs,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            n_episodes=cfg.n_episodes,
            max_episodes_rendered=cfg.max_episodes_rendered,
            videos_dir=videos_dir,
            start_seed=cfg.seed,
            max_parallel_tasks=cfg.max_parallel_tasks,
        )
    close_envs(envs)
    elapsed = time.time() - start

    suite_dir = Path(cfg.output_dir) / suite
    suite_dir.mkdir(parents=True, exist_ok=True)
    with (suite_dir / "eval_info.json").open("w") as f:
        json.dump(info, f, indent=2)
    _dump_episode_steps_csv(info.get("per_task", []), suite_dir / "episode_steps.csv")

    summary = _suite_summary(info["overall"], cfg.n_episodes)
    summary["eval_s"] = elapsed
    logging.info(f"[{suite}] done in {elapsed:.1f}s: {summary}")
    return summary


@draccus.wrap()
def main(cfg: ThroughputEvalConfig):
    init_logging()
    register_third_party_plugins()
    logging.info(pformat(cfg))

    device = get_safe_torch_device(cfg.device, log=True) if cfg.device else get_safe_torch_device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    # Build policy & shared (env-agnostic) processors once. LIBERO suites share the same observation /
    # action layout, so a bootstrap env config from the first suite is enough to resolve dataset features.
    bootstrap_env_cfg = LiberoEnv(task=cfg.suites[0])

    policy_cfg = PreTrainedConfig.from_pretrained(cfg.policy_path)
    policy_cfg.pretrained_path = cfg.policy_path
    if cfg.device is not None:
        policy_cfg.device = cfg.device
    policy_cfg.use_amp = cfg.use_amp

    policy = make_policy(cfg=policy_cfg, env_cfg=bootstrap_env_cfg)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=cfg.policy_path,
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
    )

    out_root = Path(cfg.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, dict] = {}
    for suite in cfg.suites:
        summaries[suite] = run_one_suite(suite, cfg, policy, preprocessor, postprocessor, device)

    with (out_root / "libero_throughput.json").open("w") as f:
        json.dump(
            {
                "policy_path": cfg.policy_path,
                "n_episodes_per_suite": cfg.n_episodes,
                "summaries": summaries,
            },
            f,
            indent=2,
        )

    logging.info("=== LIBERO throughput baseline ===")
    for suite, summary in summaries.items():
        logging.info(
            f"{suite:20s} | success={summary['pc_success']:.1f}% "
            f"| avg_len={summary['avg_episode_length']:.1f} "
            f"| throughput={summary['throughput']:.3f}"
        )


if __name__ == "__main__":
    main()
