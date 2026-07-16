"""Policy gradient trading agent (REINFORCE).

This implementation is optional and contains a basic policy gradient structure.
"""
import numpy as np
from rl.agent_base import AgentBase
from rl.rl_utils import save_model, load_model

class PolicyGradientAgent(AgentBase):
    def __init__(self, state_dim: int, action_dim: int, learning_rate: float = 0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = learning_rate
        self.policy = np.ones((state_dim, action_dim)) * 0.01

    def act(self, state):
        action_probs = self._action_probabilities(state)
        return int(np.random.choice(self.action_dim, p=action_probs))

    def train(self, env, episodes: int = 50):
        for episode in range(episodes):
            state = env.reset()
            rewards = []
            states = []
            actions = []
            done = False
            while not done:
                action_probs = self._action_probabilities(state)
                action = int(np.random.choice(self.action_dim, p=action_probs))
                next_state, reward, done, _ = env.step(action)
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                state = next_state
            self._update_policy(states, actions, rewards)

    def save_model(self, path: str):
        save_model(self.policy, path)

    def load_model(self, path: str):
        self.policy = load_model(path)

    def _action_probabilities(self, state):
        logits = np.dot(state, self.policy)
        exp = np.exp(logits - np.max(logits))
        return exp / exp.sum()

    def _update_policy(self, states, actions, rewards):
        discounted = self._discount_rewards(rewards)
        for state, action, reward in zip(states, actions, discounted):
            probs = self._action_probabilities(state)
            probs[action] += self.lr * reward
            self.policy += np.outer(state, probs - probs.mean())

    def _discount_rewards(self, rewards, gamma=0.99):
        discounted = np.zeros_like(rewards, dtype=float)
        running = 0
        for t in reversed(range(len(rewards))):
            running = running * gamma + rewards[t]
            discounted[t] = running
        return discounted
