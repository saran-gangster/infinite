# infinite/train/trainer/__init__.py
from hydra.core.config_store import ConfigStore

from .base import Trainer
from .grpo import GRPOTrainer
from config.types import VerifiersEnvConfig

# CRITICAL: Registers the VerifiersEnvConfig dataclass with Hydra.
cs = ConfigStore.instance()
cs.store(
    name="verifiers_env_schema", 
    node=VerifiersEnvConfig, 
    group="rollout/verifiers_envs"
)

__all__ = ["Trainer", "GRPOTrainer"]