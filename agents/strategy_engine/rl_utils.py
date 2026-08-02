"""Utility helpers for RL agents."""
from __future__ import annotations

import os
import random
from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np
import torch
from torch import nn


class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer: Deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        state_array = np.asarray(state, dtype=np.float32)
        next_state_array = np.asarray(next_state, dtype=np.float32)
        self.buffer.append((state_array, action, float(reward), next_state_array, bool(done)))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states).astype(np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states).astype(np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


def save_model(model: nn.Module, path: str) -> None:
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, path)


def load_model(model: nn.Module, path: str, device: Optional[torch.device] = None) -> None:
    checkpoint = torch.load(path, map_location=device or torch.device("cpu"))
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
