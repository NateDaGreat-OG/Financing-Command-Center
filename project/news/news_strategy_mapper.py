def map_news_to_strategy(features, reaction):
    sentiment = features["sentiment"]
    event_type = features["event_type"]

    if event_type == "earnings":
        if sentiment > 0:
            return "trend_continuation"
        else:
            return "volatility_compression"

    if event_type == "guidance":
        return "liquidity_window"

    if event_type == "m&a":
        return "scalping_premarket"

    return "magic_formula"
