"""Tests for the --yaml_config CLI feature in lerobot.configs.parser.

This feature allows users to specify a YAML file as the base configuration
via `--yaml_config=path/to/config.yaml`, with CLI args overriding YAML values.

Tests use both synthetic configs (unit) and real experiments/ YAML files (integration).
"""

import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from lerobot.configs.parser import wrap

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"


# ---------------------------------------------------------------------------
# Synthetic (unit) tests — fast, no heavy imports
# ---------------------------------------------------------------------------

@dataclass
class InnerConfig:
    value: int = 0
    name: str = "default"


@dataclass
class SimpleConfig:
    batch_size: int = 8
    steps: int = 100
    job_name: str = "test"
    inner: InnerConfig = field(default_factory=InnerConfig)


@pytest.fixture
def yaml_config_file(tmp_path):
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(
        textwrap.dedent("""\
            batch_size: 32
            steps: 5000
            job_name: yaml_job
            inner:
              value: 42
              name: from_yaml
        """)
    )
    return config_file


@pytest.fixture
def minimal_yaml_config(tmp_path):
    config_file = tmp_path / "minimal.yaml"
    config_file.write_text(
        textwrap.dedent("""\
            batch_size: 16
        """)
    )
    return config_file


class TestYamlConfigUnit:
    """Unit tests with synthetic dataclass configs."""

    def test_yaml_config_loads_values(self, yaml_config_file):
        @wrap()
        def fn(cfg: SimpleConfig):
            return cfg

        sys.argv = ["script.py", f"--yaml_config={yaml_config_file}"]
        cfg = fn()

        assert cfg.batch_size == 32
        assert cfg.steps == 5000
        assert cfg.job_name == "yaml_job"
        assert cfg.inner.value == 42
        assert cfg.inner.name == "from_yaml"

    def test_cli_overrides_yaml(self, yaml_config_file):
        @wrap()
        def fn(cfg: SimpleConfig):
            return cfg

        sys.argv = [
            "script.py",
            f"--yaml_config={yaml_config_file}",
            "--batch_size=64",
            "--job_name=cli_override",
        ]
        cfg = fn()

        assert cfg.batch_size == 64
        assert cfg.job_name == "cli_override"
        assert cfg.steps == 5000  # from YAML
        assert cfg.inner.value == 42  # from YAML

    def test_partial_yaml_uses_defaults(self, minimal_yaml_config):
        @wrap()
        def fn(cfg: SimpleConfig):
            return cfg

        sys.argv = ["script.py", f"--yaml_config={minimal_yaml_config}"]
        cfg = fn()

        assert cfg.batch_size == 16
        assert cfg.steps == 100  # default
        assert cfg.job_name == "test"  # default

    def test_no_yaml_config_uses_defaults(self):
        @wrap()
        def fn(cfg: SimpleConfig):
            return cfg

        sys.argv = ["script.py"]
        cfg = fn()

        assert cfg.batch_size == 8
        assert cfg.steps == 100

    def test_yaml_config_stripped_from_cli_args(self, yaml_config_file):
        @wrap()
        def fn(cfg: SimpleConfig):
            return cfg

        sys.argv = ["script.py", f"--yaml_config={yaml_config_file}"]
        cfg = fn()
        assert cfg.batch_size == 32


# ---------------------------------------------------------------------------
# Integration tests — parse real experiments/ YAML into TrainPipelineConfig
# ---------------------------------------------------------------------------

class TestYamlConfigExperiments:
    """Integration tests using real experiments/ YAML files."""

    @staticmethod
    def _import_train_config():
        """Import TrainPipelineConfig and ensure pi05 policy is registered."""
        # Importing the pi05 config module triggers @register_subclass("pi05")
        import lerobot.policies.pi05.configuration_pi05  # noqa: F401
        from lerobot.configs.train import TrainPipelineConfig
        return TrainPipelineConfig

    @pytest.mark.parametrize(
        "yaml_filename",
        [
            "pi05_expert_so101_table_cleanup.yaml",
            "pi05_lora_so101_table_cleanup.yaml",
        ],
    )
    def test_experiment_yaml_parses(self, yaml_filename):
        """Each experiments/ YAML should parse into a valid TrainPipelineConfig."""
        yaml_path = EXPERIMENTS_DIR / yaml_filename
        if not yaml_path.exists():
            pytest.skip(f"{yaml_path} not found")

        TrainPipelineConfig = self._import_train_config()

        @wrap()
        def fn(cfg: TrainPipelineConfig):
            return cfg

        sys.argv = ["script.py", f"--yaml_config={yaml_path}"]
        cfg = fn()

        assert isinstance(cfg, TrainPipelineConfig)
        assert cfg.dataset.repo_id == "Atticuxz/so101-table-cleanup"
        assert cfg.job_name is not None

    def test_expert_yaml_values(self):
        """Verify specific values from the expert training YAML."""
        yaml_path = EXPERIMENTS_DIR / "pi05_expert_so101_table_cleanup.yaml"
        if not yaml_path.exists():
            pytest.skip(f"{yaml_path} not found")

        TrainPipelineConfig = self._import_train_config()

        @wrap()
        def fn(cfg: TrainPipelineConfig):
            return cfg

        sys.argv = ["script.py", f"--yaml_config={yaml_path}"]
        cfg = fn()

        assert cfg.batch_size == 4
        assert cfg.steps == 8000
        assert cfg.save_freq == 2000
        assert cfg.eval_freq == 0
        assert cfg.job_name == "pi05_expert_so101"
        assert cfg.wandb.enable is True
        assert cfg.wandb.project == "pi05_so101"
        # Policy-specific fields
        assert cfg.policy is not None
        assert cfg.policy.pretrained_path == Path("lerobot/pi05_base")
        assert cfg.policy.dtype == "bfloat16"
        assert cfg.policy.gradient_checkpointing is True

    def test_lora_yaml_has_peft(self):
        """Verify the LoRA YAML correctly populates the peft config."""
        yaml_path = EXPERIMENTS_DIR / "pi05_lora_so101_table_cleanup.yaml"
        if not yaml_path.exists():
            pytest.skip(f"{yaml_path} not found")

        TrainPipelineConfig = self._import_train_config()

        @wrap()
        def fn(cfg: TrainPipelineConfig):
            return cfg

        sys.argv = ["script.py", f"--yaml_config={yaml_path}"]
        cfg = fn()

        assert cfg.peft is not None
        method_type = cfg.peft.method_type
        assert (method_type == "LORA") or (hasattr(method_type, "value") and method_type.value == "LORA")
        assert cfg.peft.r == 16
        assert cfg.batch_size == 4
        assert cfg.steps == 5000

    def test_cli_overrides_experiment_yaml(self):
        """CLI args should override values from experiments/ YAML."""
        yaml_path = EXPERIMENTS_DIR / "pi05_expert_so101_table_cleanup.yaml"
        if not yaml_path.exists():
            pytest.skip(f"{yaml_path} not found")

        TrainPipelineConfig = self._import_train_config()

        @wrap()
        def fn(cfg: TrainPipelineConfig):
            return cfg

        sys.argv = [
            "script.py",
            f"--yaml_config={yaml_path}",
            "--batch_size=16",
            "--steps=100",
        ]
        cfg = fn()

        assert cfg.batch_size == 16  # CLI override
        assert cfg.steps == 100  # CLI override
        assert cfg.job_name == "pi05_expert_so101"  # from YAML
        assert cfg.save_freq == 2000  # from YAML
