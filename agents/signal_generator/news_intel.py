from .news_scraper import fetch_news
from .news_aggregator import aggregate_by_company
from .news_features import extract_features
from .news_strategy_mapper import map_news_to_strategy
from .news_rl_refiner import refine_strategy_with_rl

def analyze_news(ticker_list, performance_history):
    headlines = fetch_news()
    grouped = aggregate_by_company(headlines, ticker_list)

    results = {}

    for ticker, news_items in grouped.items():
        enriched = []
        for h in news_items:
            features = extract_features(h)
            base_strategy = map_news_to_strategy(features, reaction=None)
            refined = refine_strategy_with_rl(ticker, base_strategy, performance_history)

            enriched.append({
                "title": h["title"],
                "sentiment": features["sentiment"],
                "event_type": features["event_type"],
                "strategy": refined
            })

        results[ticker] = enriched

    return results
