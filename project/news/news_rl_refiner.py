def refine_strategy_with_rl(ticker, strategy, performance_history):
    perf = performance_history.get((ticker, strategy), 0.0)
    # Combine base strategy with RL reward
    return strategy if perf >= 0 else "volatility_compression"
