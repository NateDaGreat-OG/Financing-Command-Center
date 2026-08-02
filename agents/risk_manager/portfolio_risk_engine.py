"""Portfolio-level risk engine.

Provides correlation analysis, beta exposure, volatility targeting,
dynamic leverage, portfolio drawdown enforcement, and risk budget allocation
across a multi-strategy, multi-symbol portfolio.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_CORR_WINDOW = 60
_DEFAULT_VOL_WINDOW = 20
_DEFAULT_TARGET_VOL = 0.15       # 15 % annualised portfolio vol
_DEFAULT_MAX_LEVERAGE = 2.0
_DEFAULT_MAX_PORTFOLIO_DD = 0.20  # 20 %
_DEFAULT_MAX_BETA = 1.5


class PortfolioRiskEngine:
    """Portfolio-level risk, correlation, beta, and volatility-targeting engine.

    Parameters
    ----------
    config:
        Application configuration dict.  Recognised keys:

        ``TARGET_VOL`` (float, default 0.15) – annualised vol target.
        ``MAX_LEVERAGE`` (float, default 2.0) – maximum allowed gross leverage.
        ``MAX_PORTFOLIO_DRAWDOWN`` (float, default 0.20) – hard drawdown ceiling.
        ``MAX_BETA`` (float, default 1.5) – maximum net market beta.
        ``CORR_WINDOW`` (int, default 60) – look-back for correlation matrix.
        ``VOL_WINDOW`` (int, default 20) – look-back for realised vol.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        self.target_vol = float(cfg.get("TARGET_VOL", _DEFAULT_TARGET_VOL))
        self.max_leverage = float(cfg.get("MAX_LEVERAGE", _DEFAULT_MAX_LEVERAGE))
        self.max_portfolio_drawdown = float(cfg.get("MAX_PORTFOLIO_DRAWDOWN", _DEFAULT_MAX_PORTFOLIO_DD))
        self.max_beta = float(cfg.get("MAX_BETA", _DEFAULT_MAX_BETA))
        self.corr_window = int(cfg.get("CORR_WINDOW", _DEFAULT_CORR_WINDOW))
        self.vol_window = int(cfg.get("VOL_WINDOW", _DEFAULT_VOL_WINDOW))

    # ------------------------------------------------------------------
    # Correlation
    # ------------------------------------------------------------------

    def correlation_matrix(self, price_map: Dict[str, pd.Series]) -> pd.DataFrame:
        """Return a pairwise correlation matrix from a dict of price series.

        Parameters
        ----------
        price_map:
            Mapping of ``symbol -> pd.Series`` of closing prices (aligned index).
        """
        if not price_map:
            return pd.DataFrame()
        df = pd.DataFrame(price_map).dropna(how="all")
        returns = df.pct_change().dropna()
        if returns.empty or len(returns) < 2:
            return pd.DataFrame(np.eye(len(price_map)), index=list(price_map), columns=list(price_map))
        window = min(self.corr_window, len(returns))
        return returns.tail(window).corr()

    # ------------------------------------------------------------------
    # Beta
    # ------------------------------------------------------------------

    def portfolio_beta(
        self,
        position_returns: Dict[str, pd.Series],
        benchmark_returns: pd.Series,
    ) -> Dict[str, float]:
        """Compute per-position and aggregate portfolio beta vs a benchmark.

        Parameters
        ----------
        position_returns:
            Mapping of ``symbol -> return series``.
        benchmark_returns:
            Market benchmark return series (same frequency).

        Returns
        -------
        Dict with ``"betas"`` (per-symbol) and ``"portfolio_beta"`` (weighted average).
        """
        betas: Dict[str, float] = {}
        bench = benchmark_returns.dropna()
        for symbol, ret in position_returns.items():
            aligned = ret.dropna().align(bench, join="inner")[0]
            b_aligned = bench.align(ret.dropna(), join="inner")[0]
            if len(aligned) < 5 or b_aligned.std() == 0:
                betas[symbol] = 1.0
                continue
            cov = float(np.cov(aligned, b_aligned)[0, 1])
            var = float(b_aligned.var())
            betas[symbol] = round(cov / var, 4) if var > 0 else 1.0

        n = max(len(betas), 1)
        portfolio_beta = round(sum(betas.values()) / n, 4)
        return {"betas": betas, "portfolio_beta": portfolio_beta}

    # ------------------------------------------------------------------
    # Volatility targeting
    # ------------------------------------------------------------------

    def vol_target_scalar(
        self,
        portfolio_returns: pd.Series,
        trading_days: int = 252,
    ) -> float:
        """Return a leverage scalar that scales realised vol to the target vol.

        A value < 1 means the portfolio is over-leveraged; > 1 means it can
        increase exposure.  The result is clipped to ``[0, max_leverage]``.
        """
        ret = portfolio_returns.dropna()
        if len(ret) < 2:
            return 1.0
        realised_vol = float(ret.tail(self.vol_window).std()) * np.sqrt(trading_days)
        if realised_vol <= 0:
            return self.max_leverage
        scalar = self.target_vol / realised_vol
        return float(np.clip(scalar, 0.0, self.max_leverage))

    # ------------------------------------------------------------------
    # Dynamic leverage
    # ------------------------------------------------------------------

    def dynamic_leverage(
        self,
        portfolio_returns: pd.Series,
        cycle_state: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Compute recommended leverage based on vol target and cycle regime.

        Reduces leverage in bear / high-vol / risk-off regimes.
        """
        cycle_state = cycle_state or {}
        base = self.vol_target_scalar(portfolio_returns)

        trend = cycle_state.get("trend", "sideways")
        volatility = cycle_state.get("volatility", "low")
        macro = cycle_state.get("macro", "risk_on")

        if trend == "bear":
            base *= 0.70
        if volatility == "high":
            base *= 0.80
        if macro == "risk_off":
            base *= 0.85

        return float(round(np.clip(base, 0.0, self.max_leverage), 4))

    # ------------------------------------------------------------------
    # Drawdown checks
    # ------------------------------------------------------------------

    def current_drawdown(self, equity_curve: pd.Series) -> float:
        """Return the current drawdown fraction from the all-time high."""
        if equity_curve.empty:
            return 0.0
        peak = equity_curve.cummax()
        dd = (equity_curve - peak) / peak.replace(0, np.nan)
        return float(abs(dd.iloc[-1]))

    def drawdown_exceeded(self, equity_curve: pd.Series) -> bool:
        """Return True when the current drawdown breaches the max threshold."""
        return self.current_drawdown(equity_curve) >= self.max_portfolio_drawdown

    # ------------------------------------------------------------------
    # Risk budget
    # ------------------------------------------------------------------

    def allocate_risk_budget(
        self,
        symbols: List[str],
        volatilities: Dict[str, float],
        capital: float,
    ) -> Dict[str, float]:
        """Equal-risk-contribution allocation given per-symbol volatility.

        Returns a mapping of ``symbol -> dollar risk budget``.
        """
        if not symbols or capital <= 0:
            return {}
        inv_vols = {
            sym: 1.0 / max(volatilities.get(sym, 0.01), 1e-6)
            for sym in symbols
        }
        total = sum(inv_vols.values())
        return {
            sym: round(capital * (inv_vols[sym] / total), 2)
            for sym in symbols
        }

    # ------------------------------------------------------------------
    # Exposure caps
    # ------------------------------------------------------------------

    def cap_position_sizes(
        self,
        signals: List[Dict[str, Any]],
        capital: float,
        max_single_position_pct: float = 0.20,
    ) -> List[Dict[str, Any]]:
        """Clip each signal's ``size`` so no single position exceeds the cap."""
        capped = []
        for sig in signals:
            entry = float(sig.get("entry_price", 0.0))
            size = float(sig.get("size", 0.0))
            if entry > 0 and capital > 0:
                max_shares = (capital * max_single_position_pct) / entry
                size = min(size, max_shares)
            capped.append({**sig, "size": round(size, 4)})
        return capped

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def diagnostics(
        self,
        price_map: Dict[str, pd.Series],
        portfolio_returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        equity_curve: Optional[pd.Series] = None,
        cycle_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Return a unified diagnostics dict for the portfolio risk state."""
        corr = self.correlation_matrix(price_map)
        lev = self.dynamic_leverage(portfolio_returns, cycle_state)
        dd = self.current_drawdown(equity_curve) if equity_curve is not None else 0.0

        beta_info: Dict[str, Any] = {}
        if benchmark_returns is not None:
            position_rets = {sym: s.pct_change().dropna() for sym, s in price_map.items()}
            beta_info = self.portfolio_beta(position_rets, benchmark_returns.pct_change().dropna())

        return {
            "correlation_matrix": corr.round(4).to_dict() if not corr.empty else {},
            "dynamic_leverage": lev,
            "current_drawdown": round(dd, 4),
            "drawdown_breached": dd >= self.max_portfolio_drawdown,
            "beta": beta_info,
            "target_vol": self.target_vol,
            "max_portfolio_drawdown": self.max_portfolio_drawdown,
        }
