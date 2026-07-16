from flask import Blueprint, abort, render_template
from financial_command_center.models import MetricsDailyStrategy, Strategy, Trade, Ticker

strategy_bp = Blueprint("strategy", __name__, url_prefix="/strategy")

@strategy_bp.route("/<int:strategy_id>")
def strategy_detail(strategy_id):
    strategy = Strategy.query.get(strategy_id)
    if strategy is None:
        abort(404)

    tickers = (
        Ticker.query.join(Trade, Ticker.id == Trade.ticker_id)
        .filter(Trade.strategy_id == strategy_id)
        .distinct()
        .all()
    )
    metrics = MetricsDailyStrategy.query.filter_by(strategy_id=strategy_id).order_by(MetricsDailyStrategy.date.desc()).all()
    return render_template("strategy.html", strategy=strategy, tickers=tickers, metrics=metrics)

@strategy_bp.route("/<int:strategy_id>/trades")
def strategy_trades(strategy_id):
    strategy = Strategy.query.get(strategy_id)
    if strategy is None:
        abort(404)

    trades = Trade.query.filter_by(strategy_id=strategy_id).order_by(Trade.timestamp_entry.desc()).all()
    return render_template("trades.html", strategy=strategy, trades=trades)
