from flask import Blueprint, render_template
from financial_command_center.models import NewsItem, Ticker

news_bp = Blueprint("news", __name__, url_prefix="/news")

@news_bp.route("")
def news():
    news_items = NewsItem.query.order_by(NewsItem.published_at.desc()).all()
    ticker_news = {}
    global_news = []

    for item in news_items:
        if item.ticker is not None:
            ticker_news.setdefault(item.ticker.symbol, []).append(item)
        else:
            global_news.append(item)

    tickers = Ticker.query.order_by(Ticker.symbol).all()
    return render_template("news.html", ticker_news=ticker_news, global_news=global_news, tickers=tickers)
