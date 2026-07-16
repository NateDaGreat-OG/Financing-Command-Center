"""Abstract base class for reinforcement learning trading agents."""
from abc import ABC, abstractmethod

class AgentBase(ABC):
    @abstractmethod
    def act(self, state):
        raise NotImplementedError

    @abstractmethod
    def train(self, env, episodes: int):
        raise NotImplementedError

    @abstractmethod
    def save_model(self, path: str):
        raise NotImplementedError

    @abstractmethod
    def load_model(self, path: str):
        raise NotImplementedError
