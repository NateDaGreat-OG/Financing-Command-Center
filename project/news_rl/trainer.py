# project/news_rl/trainer.py

from .env import NewsStrategyEnv
from .agent import DQNAgent

def train_rl_agent(backtester, historical_news, strategy_list, episodes=50):
    env = NewsStrategyEnv(backtester, historical_news)
    agent = DQNAgent(state_size=3, action_size=len(strategy_list))

    for ep in range(episodes):
        state = env.reset()
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state

        agent.replay()

    return agent
