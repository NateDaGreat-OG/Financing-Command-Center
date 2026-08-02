import math

class RiskManager:
    def __init__(self, config):
        self.max_risk_per_trade = config.get("MAX_RISK_PER_TRADE", 0.01)
        self.max_concurrent_positions = config.get("MAX_CONCURRENT_POSITIONS", 5)

    def position_size(self, capital: float, entry_price: float, stop_price: float) -> float:
        risk_amount = capital * self.max_risk_per_trade
        risk_per_share = abs(entry_price - stop_price)
        if risk_per_share <= 0:
            return 0.0
        return math.floor(risk_amount / risk_per_share)

    def stop_loss_target(self, entry_price: float, volatility: float) -> dict:
        stop = entry_price - volatility if entry_price > 0 else 0
        take_profit = entry_price + volatility * 2
        return {"stop": stop, "target": take_profit}

    def can_open_trade(self, current_positions: int) -> bool:
        return current_positions < self.max_concurrent_positions
