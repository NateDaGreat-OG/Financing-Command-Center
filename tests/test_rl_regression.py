"""Regression tests for the reinforcement learning subsystem."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch

# Add the project package path so the RL modules can be imported directly.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "project"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from rl.dqn_agent import DQNAgent
from rl.rl_utils import ReplayBuffer
from rl.trading_env import TradingEnv


def make_dummy_data(num_rows: int = 100) -> pd.DataFrame:
    """Create realistic synthetic OHLCV data for RL regression tests."""
    dates = pd.date_range(start="2024-01-01", periods=num_rows, freq="D")
    close = np.linspace(100.0, 120.0, num_rows) + np.random.normal(0.0, 0.5, num_rows)
    open_ = close + np.random.normal(0.0, 0.3, num_rows)
    high = np.maximum(open_, close) + np.abs(np.random.normal(0.5, 0.2, num_rows))
    low = np.minimum(open_, close) - np.abs(np.random.normal(0.5, 0.2, num_rows))
    volume = np.random.randint(100_000, 300_000, num_rows).astype(float)

    return pd.DataFrame(
        {
            "t": dates,
            "o": open_.round(2),
            "h": high.round(2),
            "l": low.round(2),
            "c": close.round(2),
            "v": volume,
        }
    )


def test_trading_env_state_shape():
    """TradingEnv.reset returns a state vector including 14 features and 3 mask values."""
    data = make_dummy_data(50)
    df = data.rename(columns={"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}).set_index("timestamp")
    env = TradingEnv(data=df, capital=100_000.0)

    state = env.reset()

    assert isinstance(state, np.ndarray), "Environment state must be a NumPy array"
    assert state.shape == (17,), "State must contain 14 features plus 3 action mask values"
    assert state.dtype == np.float32, "State dtype must be float32"
    assert np.all(np.isfinite(state)), "State must not contain inf or NaN values"


def test_dqn_agent_action_masking():
    """DQNAgent.act must return a valid masked action and handle action mask extraction."""
    data = make_dummy_data(50)
    df = data.rename(columns={"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}).set_index("timestamp")
    env = TradingEnv(data=df, capital=100_000.0)
    state = env.reset()

    agent = DQNAgent(state_dim=17, action_dim=3)
    action = agent.act(state)

    assert action in {0, 1, 2}, "Action must be a valid index for the action space"

    # Ensure explicit masking also works.
    mask = np.array([1.0, 0.0, 1.0], dtype=np.float32)
    action = agent.act(state, action_mask=mask)
    assert action in {0, 1, 2}, "Explicit action mask should not crash agent.act"


def test_replay_buffer_shape():
    """ReplayBuffer must return correctly shaped batches including full states with masks."""
    buffer = ReplayBuffer(capacity=100)
    dummy_state = np.arange(17, dtype=np.float32)
    for idx in range(50):
        next_state = dummy_state + idx + 1.0
        buffer.push(dummy_state, action=idx % 3, reward=float(idx) * 0.1, next_state=next_state, done=(idx % 10 == 0))

    states, actions, rewards, next_states, dones = buffer.sample(32)

    assert states.shape == (32, 17)
    assert next_states.shape == (32, 17)
    assert actions.shape == (32,)
    assert rewards.shape == (32,)
    assert dones.shape == (32,)
    assert states.dtype == np.float32
    assert next_states.dtype == np.float32
    assert actions.dtype == np.int64
    assert rewards.dtype == np.float32
    assert dones.dtype == np.float32


def test_single_episode_training():
    """DQNAgent train loop must run for one episode and return structured metrics."""
    data = make_dummy_data(100)
    df = data.rename(columns={"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}).set_index("timestamp")
    env = TradingEnv(data=df, capital=100_000.0)
    agent = DQNAgent(state_dim=17, action_dim=3, batch_size=16)

    results = agent.train(env, episodes=1, early_stop_patience=2, checkpoint_interval=0)

    assert isinstance(results, dict)
    assert results["episodes"] == 1
    assert "best_reward" in results
    assert "history" in results
    assert isinstance(results["history"], list)
    assert len(results["history"]) >= 1
    assert all(key in results["history"][0] for key in ["episode", "reward", "avg_loss", "epsilon", "steps", "final_equity"])


def test_save_load_cycle(tmp_path):
    """Saving and loading a trained agent must preserve model parameters."""
    data = make_dummy_data(100)
    df = data.rename(columns={"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}).set_index("timestamp")
    env = TradingEnv(data=df, capital=100_000.0)
    agent = DQNAgent(state_dim=17, action_dim=3, batch_size=16)

    agent.train(env, episodes=1, early_stop_patience=2, checkpoint_interval=0)

    model_path = tmp_path / "dqn_test_model.pth"
    agent.save_model(str(model_path))

    loaded_agent = DQNAgent(state_dim=17, action_dim=3)
    loaded_agent.load_model(str(model_path))

    for param_name, param_value in agent.policy_net.state_dict().items():
        loaded_value = loaded_agent.policy_net.state_dict()[param_name]
        assert torch.allclose(param_value, loaded_value), f"Parameter {param_name} did not match after save/load"
