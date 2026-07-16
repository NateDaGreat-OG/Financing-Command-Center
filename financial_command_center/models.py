from datetime import datetime
from financial_command_center.app import db

class Strategy(db.Model):
    __tablename__ = "strategies"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(128), nullable=False)
    description = db.Column(db.Text, nullable=True)
    active = db.Column(db.Boolean, default=True)

    portfolios = db.relationship("Portfolio", back_populates="strategy", cascade="all, delete-orphan")
    positions = db.relationship("Position", back_populates="strategy", cascade="all, delete-orphan")
    trades = db.relationship("Trade", back_populates="strategy", cascade="all, delete-orphan")
    daily_metrics = db.relationship("MetricsDailyStrategy", back_populates="strategy", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Strategy {self.name}>"

class Portfolio(db.Model):
    __tablename__ = "portfolios"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(128), nullable=False)
    description = db.Column(db.Text, nullable=True)
    strategy_id = db.Column(db.Integer, db.ForeignKey("strategies.id"), nullable=False)

    strategy = db.relationship("Strategy", back_populates="portfolios")
    positions = db.relationship("Position", back_populates="portfolio", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Portfolio {self.name}>"

class Ticker(db.Model):
    __tablename__ = "tickers"

    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(16), unique=True, nullable=False)
    name = db.Column(db.String(128), nullable=False)
    sector = db.Column(db.String(128), nullable=True)

    positions = db.relationship("Position", back_populates="ticker", cascade="all, delete-orphan")
    trades = db.relationship("Trade", back_populates="ticker", cascade="all, delete-orphan")
    news_items = db.relationship("NewsItem", back_populates="ticker", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Ticker {self.symbol}>"

class Position(db.Model):
    __tablename__ = "positions"

    id = db.Column(db.Integer, primary_key=True)
    portfolio_id = db.Column(db.Integer, db.ForeignKey("portfolios.id"), nullable=False)
    ticker_id = db.Column(db.Integer, db.ForeignKey("tickers.id"), nullable=False)
    strategy_id = db.Column(db.Integer, db.ForeignKey("strategies.id"), nullable=False)
    open_timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    close_timestamp = db.Column(db.DateTime, nullable=True)
    entry_price = db.Column(db.Float, nullable=False)
    exit_price = db.Column(db.Float, nullable=True)
    size = db.Column(db.Float, nullable=False)
    side = db.Column(db.String(8), nullable=False)
    tags = db.Column(db.String(256), nullable=True)

    portfolio = db.relationship("Portfolio", back_populates="positions")
    ticker = db.relationship("Ticker", back_populates="positions")
    strategy = db.relationship("Strategy", back_populates="positions")
    trades = db.relationship("Trade", back_populates="position", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Position {self.id} {self.ticker.symbol}>",

class Trade(db.Model):
    __tablename__ = "trades"

    id = db.Column(db.Integer, primary_key=True)
    position_id = db.Column(db.Integer, db.ForeignKey("positions.id"), nullable=False)
    ticker_id = db.Column(db.Integer, db.ForeignKey("tickers.id"), nullable=False)
    strategy_id = db.Column(db.Integer, db.ForeignKey("strategies.id"), nullable=False)
    timestamp_entry = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    timestamp_exit = db.Column(db.DateTime, nullable=True)
    entry_price = db.Column(db.Float, nullable=False)
    exit_price = db.Column(db.Float, nullable=True)
    size = db.Column(db.Float, nullable=False)
    side = db.Column(db.String(8), nullable=False)
    gross_pnl = db.Column(db.Float, nullable=True)
    fees = db.Column(db.Float, nullable=True)
    slippage = db.Column(db.Float, nullable=True)
    net_pnl = db.Column(db.Float, nullable=True)
    holding_period_seconds = db.Column(db.Integer, nullable=True)
    is_short_term = db.Column(db.Boolean, default=False)
    is_long_term = db.Column(db.Boolean, default=False)
    rule_break_flag = db.Column(db.Boolean, default=False)
    notes = db.Column(db.Text, nullable=True)

    position = db.relationship("Position", back_populates="trades")
    ticker = db.relationship("Ticker", back_populates="trades")
    strategy = db.relationship("Strategy", back_populates="trades")

    def __repr__(self):
        return f"<Trade {self.id} {self.ticker.symbol}>"

class MetricsDailyStrategy(db.Model):
    __tablename__ = "metrics_daily_strategy"

    id = db.Column(db.Integer, primary_key=True)
    strategy_id = db.Column(db.Integer, db.ForeignKey("strategies.id"), nullable=False)
    date = db.Column(db.Date, nullable=False)
    trade_count = db.Column(db.Integer, default=0)
    gross_pnl = db.Column(db.Float, default=0.0)
    net_pnl = db.Column(db.Float, default=0.0)
    win_rate = db.Column(db.Float, default=0.0)
    profit_factor = db.Column(db.Float, default=0.0)
    max_drawdown = db.Column(db.Float, default=0.0)
    short_term_realized = db.Column(db.Float, default=0.0)
    long_term_realized = db.Column(db.Float, default=0.0)
    fees_total = db.Column(db.Float, default=0.0)
    slippage_total = db.Column(db.Float, default=0.0)

    strategy = db.relationship("Strategy", back_populates="daily_metrics")

    def __repr__(self):
        return f"<MetricsDailyStrategy {self.strategy_id} {self.date}>"

class TaxSummary(db.Model):
    __tablename__ = "tax_summaries"

    id = db.Column(db.Integer, primary_key=True)
    year = db.Column(db.Integer, nullable=False)
    short_term_gains = db.Column(db.Float, default=0.0)
    short_term_losses = db.Column(db.Float, default=0.0)
    long_term_gains = db.Column(db.Float, default=0.0)
    long_term_losses = db.Column(db.Float, default=0.0)
    harvested_losses = db.Column(db.Float, default=0.0)
    estimated_tax_liability = db.Column(db.Float, default=0.0)

    def __repr__(self):
        return f"<TaxSummary {self.year}>"

class NewsItem(db.Model):
    __tablename__ = "news_items"

    id = db.Column(db.Integer, primary_key=True)
    ticker_id = db.Column(db.Integer, db.ForeignKey("tickers.id"), nullable=True)
    headline = db.Column(db.String(256), nullable=False)
    source = db.Column(db.String(128), nullable=True)
    category = db.Column(db.String(128), nullable=True)
    published_at = db.Column(db.DateTime, default=datetime.utcnow)
    url = db.Column(db.String(512), nullable=True)
    impact_tag = db.Column(db.String(64), nullable=True)

    ticker = db.relationship("Ticker", back_populates="news_items")

    def __repr__(self):
        return f"<NewsItem {self.headline[:40]}>"
