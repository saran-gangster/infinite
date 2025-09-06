# infinite/config/types.py
from dataclasses import dataclass, field
from typing import Any, Dict, List

@dataclass
class VerifiersEnvConfig:
    """Configuration for a single verifiers environment."""
    id: str = "???"
    kwargs: Dict[str, Any] = field(default_factory=dict)