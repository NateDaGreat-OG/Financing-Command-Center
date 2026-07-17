"""Tests for the AI intelligence layer.

Covers all six new modules:
  - strategy_ai_utils
  - strategy_cycle_adapter
  - strategy_rl_adapter
  - strategy_risk_adapter
  - strategy_capital_adapter
  - strategy_intelligence

Also verifies that strategy_registry new helpers work and that each
updated strategy module accepts set_params without breaking generate_signals.
"""
from __future__ import annotations

import os
import sys
import types
from typing import Any

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Add project/ to path so we can import its packages directly
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "project"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core.strategy_ai_utils import (
    compute_technical_indicators,
    compute_ensemble_signals,
    enrich_signal,
    filter_signals_by_quality,
    score_signal,
)
from core.strategy_cycle_adapter import CycleAdapter
from core.strategy_rl_adapter import RLAdapter
from core.strategy_risk_adapter import RiskAdapter
from core.strategy_capital_adapter import CapitalAdapter
from core.strategy_intelligence import StrategyIntelligence
from core.strategy_registry import (
    create_intelligence_layer,
    load_all_strategies,
    load_strategy,
    load_strategy_with_intelligence,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_ohlcv(rows: int = 60) -> pd.DataFrame:
    """Synthetic daily OHLCV DataFrame with a DatetimeIndex."""
    dates = pd.date_range("2024-01-01", periods=rows, freq="D")
    close = np.linspace(100.0, 130.0, rows) + np.random.normal(0, 0.5, rows)
    open_ = close + np.random.normal(0, 0.3, rows)
    high = np.maximum(open_, close) + np.abs(np.random.normal(0.5, 0.2, rows))
    low = np.minimum(open_, close) - np.abs(np.random.normal(0.5, 0.2, rows))
    volume = np.random.randint(100_000, 300_000, rows).astype(float)
    return pd.DataFrame(
        {"open": open_.round(2), "high": high.round(2), "low": low.round(2),
         "close": close.round(2), "volume": volume},
        index=dates,
    )


_BULL_CYCLE = {"trend": "bull", "volatility": "low", "liquidity": "expanding",
               "macro": "risk_on", "intraday": "open_drive"}
_BEAR_CYCLE = {"trend": "bear", "volatility": "high", "liquidity": "contracting",
               "macro": "risk_off", "intraday": "chop"}


# ===========================================================================
# strategy_ai_utils
# ===========================================================================

class TestComputeTechnicalIndicators:
    def test_returns_expected_columns(self):
        df = _make_ohlcv(60)
        out = compute_technical_indicators(df)
        for col in ["ema9", "ema20", "ema50", "atr", "rsi", "vwap",
                    "bb_mid", "bb_upper", "bb_lower", "bb_width",
                    "vol_avg", "vol_ratio", "momentum"]:
            assert col in out.columns, f"Missing column: {col}"

    def test_no_inf_or_nan(self):
        df = _make_ohlcv(60)
        out = compute_technical_indicators(df)
        numeric = out.select_dtypes(include=[np.number])
        assert np.all(np.isfinite(numeric.values)), "Output contains inf or NaN"

    def test_empty_input(self):
        empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        out = compute_technical_indicators(empty)
        assert out.empty


class TestScoreSignal:
    def test_score_in_range(self):
        df = _make_ohlcv(60)
        signal = {"side": "long", "entry_price": 120.0, "stop_loss": 115.0, "target": 130.0,
                  "signal_type": "trend_continuation"}
        score = score_signal(signal, df, _BULL_CYCLE)
        assert 0.0 <= score <= 1.0

    def test_bull_long_score_higher_than_bear(self):
        df = _make_ohlcv(60)
        sig = {"side": "long", "entry_price": 120.0, "stop_loss": 115.0, "target": 130.0,
               "signal_type": "trend_continuation"}
        bull_score = score_signal(sig, df, _BULL_CYCLE)
        bear_score = score_signal(sig, df, _BEAR_CYCLE)
        assert bull_score > bear_score

    def test_empty_data_returns_neutral(self):
        sig = {"side": "long"}
        score = score_signal(sig, pd.DataFrame(), {})
        assert score == 0.5


class TestFilterSignalsByQuality:
    def test_filters_below_threshold(self):
        signals = [{"ai_score": 0.3}, {"ai_score": 0.6}, {"ai_score": 0.45}]
        result = filter_signals_by_quality(signals, min_score=0.4)
        assert len(result) == 2
        assert all(s["ai_score"] >= 0.4 for s in result)

    def test_empty_list(self):
        assert filter_signals_by_quality([], 0.4) == []


class TestComputeEnsembleSignals:
    def test_deduplicates_by_symbol_side(self):
        all_sigs = {
            "strat_a": [{"symbol": "AAPL", "side": "long", "ai_score": 0.6}],
            "strat_b": [{"symbol": "AAPL", "side": "long", "ai_score": 0.8}],
        }
        result = compute_ensemble_signals(all_sigs)
        assert len(result) == 1
        assert result[0]["ai_score"] >= 0.6  # kept the higher one (possibly weight-scaled)

    def test_different_symbols_are_kept(self):
        all_sigs = {
            "strat_a": [{"symbol": "AAPL", "side": "long", "ai_score": 0.7}],
            "strat_b": [{"symbol": "TSLA", "side": "long", "ai_score": 0.7}],
        }
        result = compute_ensemble_signals(all_sigs)
        assert len(result) == 2


class TestEnrichSignal:
    def test_adds_required_keys(self):
        df = _make_ohlcv(60)
        sig = {"side": "long", "entry_price": 120.0, "stop_loss": 115.0, "target": 130.0}
        enriched = enrich_signal(sig, df, _BULL_CYCLE, source_strategy="test_strategy")
        assert "ai_score" in enriched
        assert "cycle_context" in enriched
        assert enriched["source_strategy"] == "test_strategy"
        assert 0.0 <= enriched["ai_score"] <= 1.0


# ===========================================================================
# strategy_cycle_adapter
# ===========================================================================

class TestCycleAdapter:
    def test_adapt_params_bull_trend(self):
        adapter = CycleAdapter()
        params = adapter.adapt_params("trend_continuation", cycle_state={"trend": "bull"})
        assert "ema_short" in params
        assert params["atr_mult"] == 2.5

    def test_adapt_params_unknown_strategy(self):
        adapter = CycleAdapter()
        params = adapter.adapt_params("nonexistent_strategy", base_params={"x": 1}, cycle_state={})
        assert params == {"x": 1}

    def test_adapt_signal_confidence_bull_vs_risk_off(self):
        adapter = CycleAdapter()
        bull_conf = adapter.adapt_signal_confidence({}, _BULL_CYCLE)
        bear_conf = adapter.adapt_signal_confidence({}, _BEAR_CYCLE)
        assert bull_conf > bear_conf
        assert 0.1 <= bull_conf <= 1.5

    def test_get_favored_strategies(self):
        adapter = CycleAdapter()
        favored = adapter.get_favored_strategies({"trend": "bull", "volatility": "high"})
        assert isinstance(favored, list)
        assert len(favored) > 0

    def test_scale_size_reduces_in_risk_off(self):
        adapter = CycleAdapter()
        original = 100.0
        scaled = adapter.scale_size_for_cycle(original, {"macro": "risk_off"})
        assert scaled < original


# ===========================================================================
# strategy_rl_adapter
# ===========================================================================

class TestRLAdapter:
    def test_build_env_state_returns_correct_shape(self):
        adapter = RLAdapter()
        df = _make_ohlcv(60)
        state = adapter.build_env_state(df)
        assert state is not None
        assert state.shape == (17,)
        assert state.dtype == np.float32
        assert np.all(np.isfinite(state))

    def test_build_env_state_short_data_returns_none(self):
        adapter = RLAdapter()
        df = _make_ohlcv(5)  # too short
        assert adapter.build_env_state(df) is None

    def test_hybrid_decision_pure_strategy(self):
        adapter = RLAdapter(blend_weight=0.0)
        strat_sigs = [{"side": "long", "ai_score": 0.7}]
        result = adapter.hybrid_decision(strat_sigs, {"side": "long", "ai_score": 0.8}, None)
        assert result == strat_sigs

    def test_hybrid_decision_pure_rl(self):
        adapter = RLAdapter(blend_weight=1.0)
        rl_sig = {"side": "long", "ai_score": 0.8}
        result = adapter.hybrid_decision([{"side": "long", "ai_score": 0.5}], rl_sig, None)
        assert result == [rl_sig]

    def test_hybrid_decision_blend(self):
        adapter = RLAdapter(blend_weight=0.5)
        strat_sigs = [{"side": "long", "ai_score": 0.6}]
        rl_sig = {"side": "long", "ai_score": 0.8}
        result = adapter.hybrid_decision(strat_sigs, rl_sig, pd.DataFrame())
        # Both kept (strategy + rl)
        assert len(result) == 2

    def test_adjust_size_reduces_for_high_epsilon(self):
        adapter = RLAdapter()

        class MockAgent:
            epsilon = 1.0  # full exploration

        adjusted = adapter.adjust_size_from_rl_confidence(100.0, MockAgent())
        assert adjusted < 100.0

    def test_rl_signal_none_when_agent_is_none(self):
        adapter = RLAdapter()
        state = np.zeros(17, dtype=np.float32)
        result = adapter.get_rl_signal(None, state, "AAPL", 100.0)
        assert result is None


# ===========================================================================
# strategy_risk_adapter
# ===========================================================================

class TestRiskAdapter:
    def test_size_signal_reduces_in_risk_off(self):
        adapter = RiskAdapter(config={"MAX_RISK_PER_TRADE": 0.01})
        sig = {"entry_price": 100.0, "stop_loss": 95.0}
        sized_normal = adapter.size_signal(sig, 100_000.0, cycle_state={})
        sized_risk_off = adapter.size_signal(sig, 100_000.0, cycle_state={"macro": "risk_off"})
        assert sized_risk_off["size"] <= sized_normal["size"]

    def test_size_signal_returns_at_least_1(self):
        adapter = RiskAdapter(config={"MAX_RISK_PER_TRADE": 0.0001})
        sig = {"entry_price": 100.0, "stop_loss": 99.0}
        result = adapter.size_signal(sig, 100.0)
        assert result["size"] >= 1

    def test_apply_risk_constraints_respects_slot_limit(self):
        adapter = RiskAdapter(config={"MAX_CONCURRENT_POSITIONS": 3})
        signals = [{"entry_price": 100.0, "stop_loss": 90.0, "target": 120.0}] * 5
        result = adapter.apply_risk_constraints(signals, current_positions=2, capital=100_000.0)
        assert len(result) <= 1  # only 1 slot free

    def test_apply_risk_constraints_no_slots(self):
        adapter = RiskAdapter(config={"MAX_CONCURRENT_POSITIONS": 2})
        signals = [{"entry_price": 100.0, "stop_loss": 95.0, "target": 110.0}]
        result = adapter.apply_risk_constraints(signals, current_positions=2, capital=100_000.0)
        assert result == []

    def test_compute_stop_and_target_long(self):
        adapter = RiskAdapter()
        sig = {"side": "long", "entry_price": 100.0}
        result = adapter.compute_stop_and_target(sig, atr=2.0)
        assert result["stop_loss"] < 100.0
        assert result["target"] > 100.0

    def test_compute_stop_and_target_short(self):
        adapter = RiskAdapter()
        sig = {"side": "short", "entry_price": 100.0}
        result = adapter.compute_stop_and_target(sig, atr=2.0)
        assert result["stop_loss"] > 100.0
        assert result["target"] < 100.0

    def test_compute_stop_and_target_zero_atr(self):
        adapter = RiskAdapter()
        sig = {"side": "long", "entry_price": 100.0, "stop_loss": 95.0, "target": 110.0}
        result = adapter.compute_stop_and_target(sig, atr=0.0)
        # Original values preserved when ATR is zero
        assert result["stop_loss"] == 95.0
        assert result["target"] == 110.0

    def test_kelly_fraction_bounds(self):
        adapter = RiskAdapter(config={"MAX_RISK_PER_TRADE": 0.01})
        fraction = adapter.kelly_fraction(win_rate=0.6, avg_win=1.5, avg_loss=1.0)
        assert fraction > 0
        assert fraction <= 0.03  # capped at 3 × max_risk_per_trade

    def test_rr_filter_removes_low_rr(self):
        adapter = RiskAdapter(config={"MIN_RR_RATIO": 1.5})
        signals = [
            {"entry_price": 100.0, "stop_loss": 95.0, "target": 106.0},  # R:R 1.2 → rejected
            {"entry_price": 100.0, "stop_loss": 95.0, "target": 112.0},  # R:R 2.4 → accepted
        ]
        result = adapter.apply_risk_constraints(signals, 0, 100_000.0)
        assert len(result) == 1
        assert result[0]["target"] == 112.0


# ===========================================================================
# strategy_capital_adapter
# ===========================================================================

class TestCapitalAdapter:
    def _make_allocation_map(self):
        return {
            "my_strategy": {
                "AAPL": {
                    "allocated_capital": 10_000.0,
                    "max_position_size": 5_000.0,
                    "risk_budget": 500.0,
                }
            }
        }

    def test_size_from_allocation_sets_size(self):
        adapter = CapitalAdapter(config={})
        sig = {"entry_price": 100.0, "size": 1}
        result = adapter.size_from_allocation(sig, self._make_allocation_map(), "my_strategy", "AAPL")
        assert result["size"] > 1
        assert result["allocated_capital"] == 10_000.0

    def test_size_from_allocation_no_entry_price(self):
        adapter = CapitalAdapter(config={})
        sig = {"entry_price": 0.0, "size": 10}
        result = adapter.size_from_allocation(sig, self._make_allocation_map(), "my_strategy", "AAPL")
        assert result["size"] == 10  # unchanged

    def test_enforce_exposure_limits_trims_size(self):
        adapter = CapitalAdapter(config={"MAX_POSITION_SIZE_PCT": 0.10})
        signals = [{"entry_price": 100.0, "size": 2000}]  # notional 200k > 10% of 100k
        result = adapter.enforce_exposure_limits(signals, total_capital=100_000.0)
        assert len(result) == 1
        assert result[0]["size"] < 2000

    def test_enforce_exposure_limits_drops_when_exhausted(self):
        adapter = CapitalAdapter(config={"MAX_POSITION_SIZE_PCT": 0.10})
        signals = [
            {"entry_price": 100.0, "size": 100},   # notional 10k = full budget
            {"entry_price": 100.0, "size": 10},    # no budget left
        ]
        result = adapter.enforce_exposure_limits(signals, total_capital=100_000.0)
        assert len(result) == 1

    def test_compute_kelly_size_reasonable(self):
        adapter = CapitalAdapter(config={})
        sig = {"entry_price": 100.0, "size": 10}
        result = adapter.compute_kelly_size(sig, win_rate=0.55, avg_win=1.5, avg_loss=1.0, capital=100_000.0)
        assert result["size"] > 0
        assert "kelly_fraction" in result
        assert 0 < result["kelly_fraction"] <= 0.25


# ===========================================================================
# strategy_intelligence (main layer)
# ===========================================================================

def _make_trend_strategy_module():
    """Create a minimal strategy module stub for testing."""
    mod = types.ModuleType("test_trend")
    mod.__name__ = "test_trend"

    _params = {"size": 10}

    def set_params(p):
        _params.update(p)

    def generate_signals(data):
        if data.empty or len(data) < 30:
            return []
        entry = float(data["close"].iloc[-1])
        return [{
            "side": "long",
            "entry_price": entry,
            "stop_loss": entry * 0.95,
            "target": entry * 1.10,
            "size": _params["size"],
            "signal_type": "test_trend",
        }]

    def scan_candidates(symbols):
        return [{"symbol": s} for s in symbols]

    def execute_signals(signals):
        return signals

    mod.set_params = set_params
    mod.generate_signals = generate_signals
    mod.scan_candidates = scan_candidates
    mod.execute_signals = execute_signals
    return mod


class TestStrategyIntelligence:
    def test_run_returns_expected_keys(self):
        intel = StrategyIntelligence(config={"DEFAULT_CAPITAL": 100_000.0})
        mod = _make_trend_strategy_module()
        df = _make_ohlcv(60)
        result = intel.run(mod, df, symbol="AAPL", strategy_name="test_trend",
                           cycle_state=_BULL_CYCLE)
        assert "signals" in result
        assert "cycle_state" in result
        assert "cycle_params" in result
        assert "rl_signal" in result
        assert "metadata" in result

    def test_run_without_signals_when_data_short(self):
        intel = StrategyIntelligence(config={})
        mod = _make_trend_strategy_module()
        df = _make_ohlcv(5)  # too short for strategy
        result = intel.run(mod, df, symbol="AAPL", cycle_state={})
        assert result["signals"] == []

    def test_run_with_rl_agent(self):
        from rl.dqn_agent import DQNAgent
        agent = DQNAgent(state_dim=17, action_dim=3)
        intel = StrategyIntelligence(rl_agent=agent, config={"DEFAULT_CAPITAL": 100_000.0},
                                     rl_blend_weight=0.5)
        mod = _make_trend_strategy_module()
        df = _make_ohlcv(60)
        result = intel.run(mod, df, symbol="AAPL", strategy_name="test_trend",
                           cycle_state=_BULL_CYCLE)
        # rl_signal may be None (hold action) or a dict
        assert result["rl_signal"] is None or isinstance(result["rl_signal"], dict)

    def test_run_signals_sorted_by_score(self):
        intel = StrategyIntelligence(config={}, min_signal_score=0.0)
        mod = _make_trend_strategy_module()
        df = _make_ohlcv(60)
        result = intel.run(mod, df, symbol="AAPL", strategy_name="test_trend",
                           cycle_state=_BULL_CYCLE)
        scores = [s["ai_score"] for s in result["signals"]]
        assert scores == sorted(scores, reverse=True)

    def test_run_ensemble_returns_expected_keys(self):
        intel = StrategyIntelligence(config={"DEFAULT_CAPITAL": 100_000.0})
        modules = {"test_trend": _make_trend_strategy_module()}
        df = _make_ohlcv(60)
        result = intel.run_ensemble(modules, df, symbol="AAPL", cycle_state=_BULL_CYCLE)
        assert "signals" in result
        assert "strategy_results" in result
        assert "metadata" in result

    def test_metadata_contains_correct_fields(self):
        intel = StrategyIntelligence(config={"DEFAULT_CAPITAL": 50_000.0})
        mod = _make_trend_strategy_module()
        df = _make_ohlcv(60)
        result = intel.run(mod, df, symbol="MSFT", strategy_name="test_trend",
                           cycle_state={}, capital=50_000.0)
        meta = result["metadata"]
        assert meta["symbol"] == "MSFT"
        assert meta["strategy"] == "test_trend"
        assert meta["capital"] == 50_000.0


# ===========================================================================
# strategy_registry updates
# ===========================================================================

class TestStrategyRegistry:
    def test_load_all_strategies_returns_dict(self):
        modules = load_all_strategies()
        assert isinstance(modules, dict)
        # All registered names should load successfully in the project env
        for name, mod in modules.items():
            assert mod is not None

    def test_create_intelligence_layer_returns_instance(self):
        layer = create_intelligence_layer(config={"DEFAULT_CAPITAL": 100_000.0})
        assert isinstance(layer, StrategyIntelligence)

    def test_load_strategy_with_intelligence_wraps_module(self):
        layer = create_intelligence_layer(config={})
        wrapped = load_strategy_with_intelligence("trend_continuation", layer)
        assert wrapped is not None
        assert hasattr(wrapped, "generate_signals")
        assert hasattr(wrapped, "execute_signals")
        assert hasattr(wrapped, "scan_candidates")

    def test_load_strategy_with_intelligence_unknown_returns_none(self):
        layer = create_intelligence_layer(config={})
        result = load_strategy_with_intelligence("nonexistent_strategy_xyz", layer)
        assert result is None


# ===========================================================================
# Strategy set_params compatibility
# ===========================================================================

class TestStrategySetParams:
    @pytest.mark.parametrize("strategy_name", [
        "trend_continuation",
        "volatility_compression",
        "scalping_premarket",
        "liquidity_window",
        "magic_formula",
    ])
    def test_set_params_does_not_break_import(self, strategy_name):
        mod = load_strategy(strategy_name)
        assert mod is not None
        assert hasattr(mod, "set_params"), f"{strategy_name} missing set_params"
        assert hasattr(mod, "generate_signals")
        assert hasattr(mod, "execute_signals")
        assert hasattr(mod, "scan_candidates")

    def test_trend_continuation_set_params_used(self):
        mod = load_strategy("trend_continuation")
        mod.set_params({"size": 99})
        df = _make_ohlcv(60)
        # Artificially create bull conditions so a signal fires
        df["close"] = np.linspace(100, 130, 60)
        df["open"] = df["close"] - 0.5
        df["high"] = df["close"] + 1.0
        df["low"] = df["close"] - 1.0
        df["volume"] = 200_000.0
        signals = mod.generate_signals(df)
        # If any signal fires, its size should be 99
        for sig in signals:
            assert sig["size"] == 99
        # Restore default
        mod.set_params({"size": 50})
