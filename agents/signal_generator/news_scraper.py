import feedparser

NEWS_FEEDS = [
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
    "https://www.investing.com/rss/news_25.rss",
    "https://www.marketwatch.com/rss/topstories",
    "https://www.benzinga.com/feeds/news",
    "https://www.benzinga.com/feeds/analyst-ratings",
    "https://www.benzinga.com/feeds/movers",
    "http://feeds.reuters.com/reuters/businessNews",
    "http://feeds.reuters.com/reuters/USstockNews",
    "http://feeds.reuters.com/reuters/companyNews",
    "https://stocknews.com/news/feed/"

]

def fetch_news():
    headlines = []
    for url in NEWS_FEEDS:
        feed = feedparser.parse(url)
        try:
            for entry in feed.entries:
                headlines.append({
                    "title": entry.title,
                    "link": entry.link,
                    "published": entry.get("published", None),
                    "source": url
                })
        except Exception as e:
            print(f"Error fetching news from {url}: {e}")
    return headlines
