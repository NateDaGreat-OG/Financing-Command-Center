import feedparser
from datetime import datetime

IPO_FEEDS = [
    "https://www.nasdaq.com/feed/rssoutbound?category=IPO",
    "https://www.marketwatch.com/rss/ipo",
    "https://www.investing.com/rss/news_302.rss"
]

def fetch_upcoming_ipos():
    events = []

    for url in IPO_FEEDS:
        feed = feedparser.parse(url)
        for entry in feed.entries:
            events.append({
                "type": "IPO",
                "title": entry.title,
                "link": entry.link,
                "published": entry.get("published", None)
            })

    return events
