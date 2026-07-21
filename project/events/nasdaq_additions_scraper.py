import feedparser

NASDAQ_FEEDS = [
    "https://www.nasdaq.com/feed/rssoutbound?category=PressRelease"
]

def fetch_nasdaq_additions():
    events = []

    feed = feedparser.parse(NASDAQ_FEEDS[0])
    for entry in feed.entries:
        title = str(entry.title).lower()

        if "added to nasdaq" in title or "joins nasdaq" in title:
            events.append({
                "type": "NASDAQ Addition",
                "title": entry.title,
                "link": entry.link,
                "published": entry.get("published", None)
            })

    return events
