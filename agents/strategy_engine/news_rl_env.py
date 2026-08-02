# project/news_rl/env.py

import numpy as np

class NewsStrategyEnv:
    def __init__(self, backtester, historical_news):
        self.backtester = backtester
        self.historical_news = historical_news
        self.current_index = 0

    def reset(self):
        self.current_index = 0
        return self._get_state()

    def _get_state(self):
        event = self.historical_news[self.current_index]
        return np.array([
            event["sentiment"],
            event["volatility_impact"],
            event["event_type_id"]
        ], dtype=np.float32)

    def step(self, action):
        event = self.historical_news[self.current_index]

        # Run backtest for chosen strategy
        strategy_name = event["strategy_list"][action]
        performance = self.backtester.run_strategy_on_event(event, strategy_name)

        reward = performance["sharpe"] - performance["drawdown"]

        self.current_index += 1
        done = self.current_index >= len(self.historical_news)

        next_state = self._get_state() if not done else None

        return next_state, reward, done
