"""Abstract base class for reinforcement learning trading agents."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np


class AgentBase(ABC):
    @abstractmethod
    def act(self, state: np.ndarray, action_mask: Optional[np.ndarray] = None) -> int:
        """Choose an action for the current state.

        If `action_mask` is not provided, the implementation may extract it from the end of the
        incoming state vector.
        """
        raise NotImplementedError

    @abstractmethod
    def train(self, env: Any, episodes: int = 50, **kwargs: Any) -> Dict[str, Any]:
        """Train the agent and return structured training metrics."""
        raise NotImplementedError

    @abstractmethod
    def save_model(self, path: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_model(self, path: str) -> None:
        raise NotImplementedError
