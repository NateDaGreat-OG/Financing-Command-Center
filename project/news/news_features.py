def extract_features(headline):
    text = headline["title"].lower()

    sentiment = 0
    if any(w in text for w in ["beats", "surge", "strong", "upgrade"]):
        sentiment = 1
    if any(w in text for w in ["misses", "plunge", "weak", "downgrade"]):
        sentiment = -1

    event_type = "general"
    if "earnings" in text:
        event_type = "earnings"
    elif "guidance" in text:
        event_type = "guidance"
    elif "merger" in text or "acquisition" in text:
        event_type = "m&a"
    elif "lawsuit" in text:
        event_type = "legal"

    return {
        "sentiment": sentiment,
        "event_type": event_type
    }
