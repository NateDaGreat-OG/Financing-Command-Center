"""Deep Q-network agent for trading with a PyTorch backend."""
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from rl.agent_base import AgentBase
from rl.rl_utils import ReplayBuffer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQNNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent(AgentBase):
    def __init__(self, state_dim: int, action_dim: int, lr: float = 1e-3, gamma: float = 0.99, epsilon: float = 1.0, epsilon_min: float = 0.01, epsilon_decay: float = 0.995, batch_size: int = 32, target_update: int = 10):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.memory = ReplayBuffer(capacity=10000)
        self.policy_net = DQNNetwork(state_dim, action_dim).to(DEVICE)
        self.target_net = DQNNetwork(state_dim, action_dim).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.episodes_trained = 0

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        state_tensor = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return int(q_values.argmax().item())

    def train(self, env, episodes: int = 50):
        for episode in range(episodes):
            state = env.reset()
            done = False
            while not done:
                action = self.act(state)
                next_state, reward, done, _ = env.step(action)
                self.memory.push(state, action, reward, next_state, done)
                self._replay()
                state = next_state
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            if (episode + 1) % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            self.episodes_trained += 1

    def save_model(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.policy_net.state_dict(), path)

    def load_model(self, path: str):
        self.policy_net.load_state_dict(torch.load(path, map_location=DEVICE))
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def _replay(self):
        if len(self.memory) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        state_batch = torch.tensor(np.array(states), dtype=torch.float32, device=DEVICE)
        action_batch = torch.tensor(actions, dtype=torch.long, device=DEVICE).unsqueeze(1)
        reward_batch = torch.tensor(rewards, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        next_state_batch = torch.tensor(np.array(next_states), dtype=torch.float32, device=DEVICE)
        done_batch = torch.tensor(dones, dtype=torch.float32, device=DEVICE).unsqueeze(1)

        q_values = self.policy_net(state_batch).gather(1, action_batch)
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0].unsqueeze(1)
            target_q = reward_batch + self.gamma * next_q_values * (1 - done_batch)

        loss = self.loss_fn(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
