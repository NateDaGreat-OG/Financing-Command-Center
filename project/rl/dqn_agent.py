"""Deep Q-network agent for trading with a PyTorch backend."""
from __future__ import annotations

import logging
import os
import random
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from rl.agent_base import AgentBase
from rl.rl_utils import ReplayBuffer

logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQNNetwork(nn.Module):
    def __init__(self, input_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent(AgentBase):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        batch_size: int = 32,
        tau: float = 0.005,
        max_grad_norm: float = 1.0,
        normalize_state: bool = False,
    ):
        self.state_dim = state_dim
        self.feature_dim = state_dim - 3
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.tau = tau
        self.max_grad_norm = max_grad_norm
        self.normalize_state = normalize_state
        self.memory = ReplayBuffer(capacity=10000)

        self.policy_net = DQNNetwork(self.feature_dim, action_dim).to(DEVICE)
        self.target_net = DQNNetwork(self.feature_dim, action_dim).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.episodes_trained = 0
        self.training_history: list[Dict[str, Any]] = []

    def act(self, state: np.ndarray, action_mask: Optional[np.ndarray] = None) -> int:
        state, action_mask = self._prepare_state(state, action_mask)
        valid_actions = np.where(action_mask > 0.5)[0]
        if len(valid_actions) == 0:
            return 0

        if random.random() < self.epsilon:
            return int(np.random.choice(valid_actions))

        state_tensor = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor).cpu().numpy().flatten()

        masked_q = np.full_like(q_values, -np.inf, dtype=np.float32)
        masked_q[valid_actions] = q_values[valid_actions]
        return int(np.argmax(masked_q))

    def train(
        self,
        env: Any,
        episodes: int = 50,
        early_stop_patience: int = 10,
        min_reward_delta: float = 1e-3,
        checkpoint_path: Optional[str] = None,
        checkpoint_interval: int = 10,
    ) -> Dict[str, Any]:
        best_reward = -float("inf")
        patience = 0
        best_checkpoint = None

        if checkpoint_path:
            checkpoint_dir = os.path.dirname(checkpoint_path)
            if checkpoint_dir:
                os.makedirs(checkpoint_dir, exist_ok=True)

        self.policy_net.train()
        self.target_net.eval()

        for episode in range(1, episodes + 1):
            raw_state = env.reset()
            state, action_mask = self._prepare_state(raw_state)
            episode_reward = 0.0
            episode_loss = 0.0
            step_count = 0
            done = False

            while not done:
                action = self.act(state, action_mask)
                raw_next_state, reward, done, info = env.step(action)
                next_state, next_mask = self._prepare_state(raw_next_state)

                self.memory.push(raw_state, action, reward, raw_next_state, done)
                loss = self._replay()
                if loss is not None:
                    episode_loss += loss
                episode_reward += float(reward)
                step_count += 1
                raw_state = raw_next_state
                state, action_mask = next_state, next_mask
                self._soft_update()

            self._decay_epsilon()
            avg_loss = float(episode_loss / max(step_count, 1))
            self.episodes_trained += 1
            metrics = {
                "episode": episode,
                "reward": episode_reward,
                "avg_loss": avg_loss,
                "epsilon": self.epsilon,
                "steps": step_count,
                "final_equity": float(env.equity),
            }
            self.training_history.append(metrics)
            logger.info(
                "Episode %d | reward=%.4f | avg_loss=%.4f | epsilon=%.4f | equity=%.2f | steps=%d",
                episode,
                episode_reward,
                avg_loss,
                self.epsilon,
                env.equity,
                step_count,
            )

            if checkpoint_path and checkpoint_interval > 0 and episode % checkpoint_interval == 0:
                model_checkpoint = f"{checkpoint_path}_episode_{episode}.pth"
                self.save_model(model_checkpoint)
                best_checkpoint = model_checkpoint

            if episode_reward > best_reward + min_reward_delta:
                best_reward = episode_reward
                patience = 0
            else:
                patience += 1

            if patience >= early_stop_patience:
                logger.info("Early stopping after %d episodes", episode)
                break

        return {
            "episodes": episode,
            "best_reward": best_reward,
            "history": self.training_history,
            "last_checkpoint": best_checkpoint,
        }

    def save_model(self, path: str) -> None:
        dirpath = os.path.dirname(path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        torch.save({"model_state_dict": self.policy_net.state_dict()}, path)

    def load_model(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=DEVICE)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.policy_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def _prepare_state(
        self, state: np.ndarray, action_mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        state = np.asarray(state, dtype=np.float32)
        if action_mask is None:
            if state.ndim != 1 or state.shape[0] < 3:
                raise ValueError("State must include action mask values when mask is not provided.")
            action_mask = state[-3:].astype(np.float32)
            state = state[:-3]
        else:
            action_mask = np.asarray(action_mask, dtype=np.float32)

        if self.normalize_state:
            state = self._normalize_state(state)

        return state, action_mask

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        maximum = np.maximum(np.abs(state).max(), 1.0)
        return state / maximum

    def _replay(self) -> Optional[float]:
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        state_batch = torch.tensor(states[:, :-3], dtype=torch.float32, device=DEVICE)
        next_state_batch = torch.tensor(next_states[:, :-3], dtype=torch.float32, device=DEVICE)
        next_mask_batch = torch.tensor(next_states[:, -3:], dtype=torch.float32, device=DEVICE)
        action_batch = torch.tensor(actions, dtype=torch.long, device=DEVICE).unsqueeze(1)
        reward_batch = torch.tensor(rewards, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        done_batch = torch.tensor(dones, dtype=torch.float32, device=DEVICE).unsqueeze(1)

        current_q = self.policy_net(state_batch).gather(1, action_batch)
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch)
            invalid_mask = next_mask_batch <= 0.5
            next_q_values = next_q_values.masked_fill(invalid_mask, float("-inf"))
            next_values = next_q_values.max(dim=1, keepdim=True)[0]
            next_values = next_values.masked_fill(next_values == float("-inf"), 0.0)
            target_q = reward_batch + self.gamma * next_values * (1.0 - done_batch)

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return float(loss.item())

    def _soft_update(self) -> None:
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + policy_param.data * self.tau)

    def _decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
