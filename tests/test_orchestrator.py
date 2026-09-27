"""
tests/test_orchestrator.py

Tests del orquestador de configuración (Fase 3.4): comprueba que solo se
llama a los agentes activos, que el fallo de un analista marca todo el
día/ticker como fallido sin llegar a llamar al trader, y que el caso
baseline (sin analistas) sí llega al trader.
"""

import pandas as pd
import pytest

from strategy import orchestrator


def _valid_analyst_response(role: str):
    return {
        "role": role,
        "signal": "bullish",
        "score": 0.5,
        "confidence": 0.7,
        "rationale": "x",
    }


def _valid_trader_response():
    return {"action": "buy", "size": 0.5, "confidence": 0.6, "rationale": "x"}


@pytest.fixture
def dummy_args():
    return dict(
        date=pd.Timestamp("2024-08-05"),
        ticker="NVDA",
        ohlcv_history=pd.DataFrame(),  # no se usa realmente: los agentes están mockeados
        news_items=pd.DataFrame(),
        position_state="no position (all cash)",
    )


def test_full_system_calls_all_three_and_trader(monkeypatch, dummy_args):
    monkeypatch.setattr(
        orchestrator.technical_agent, "run", lambda *a, **kw: _valid_analyst_response("technical")
    )
    monkeypatch.setattr(
        orchestrator.sentiment_agent, "run", lambda *a, **kw: _valid_analyst_response("sentiment")
    )
    monkeypatch.setattr(
        orchestrator.fundamental_agent, "run", lambda *a, **kw: _valid_analyst_response("fundamental")
    )
    monkeypatch.setattr(orchestrator.trader_agent, "run", lambda *a, **kw: _valid_trader_response())

    result = orchestrator.run_configuration(
        "cherry_pick_full_system", ["technical", "sentiment", "fundamental"], **dummy_args
    )
    assert result.status == "ok"
    assert len(result.analyst_reports) == 3
    assert result.trader_output["action"] == "buy"


def test_baseline_no_analysts_still_calls_trader(monkeypatch, dummy_args):
    monkeypatch.setattr(orchestrator.trader_agent, "run", lambda *a, **kw: _valid_trader_response())
    result = orchestrator.run_configuration(
        "cherry_pick_baseline_price_only", [], **dummy_args
    )
    assert result.status == "ok"
    assert result.analyst_reports == []
    assert result.trader_output is not None


def test_analyst_failure_marks_whole_day_failed_without_calling_trader(monkeypatch, dummy_args):
    monkeypatch.setattr(
        orchestrator.technical_agent, "run", lambda *a, **kw: _valid_analyst_response("technical")
    )
    monkeypatch.setattr(orchestrator.sentiment_agent, "run", lambda *a, **kw: None)  # falla

    trader_called = {"called": False}
    def fake_trader_run(*a, **kw):
        trader_called["called"] = True
        return _valid_trader_response()
    monkeypatch.setattr(orchestrator.trader_agent, "run", fake_trader_run)

    result = orchestrator.run_configuration(
        "cherry_pick_full_system", ["technical", "sentiment", "fundamental"], **dummy_args
    )
    assert result.status == "failed"
    assert result.failed_role == "sentiment"
    assert result.trader_output is None
    assert trader_called["called"] is False


def test_trader_failure_marks_failed_with_reports_kept(monkeypatch, dummy_args):
    monkeypatch.setattr(
        orchestrator.technical_agent, "run", lambda *a, **kw: _valid_analyst_response("technical")
    )
    monkeypatch.setattr(orchestrator.trader_agent, "run", lambda *a, **kw: None)  # falla

    result = orchestrator.run_configuration(
        "cherry_pick_no_sentiment", ["technical"], **dummy_args
    )
    assert result.status == "failed"
    assert result.failed_role == "trader"
    assert len(result.analyst_reports) == 1  # el informe técnico sí se conserva para el log


def test_load_ablation_config_returns_five_configurations():
    config = orchestrator.load_ablation_config()
    assert len(config) == 5
    assert config["cherry_pick_full_system"] == ["technical", "sentiment", "fundamental"]
    assert config["cherry_pick_baseline_price_only"] == []
    assert "cherry_pick_no_fundamental" in config
    assert "cherry_pick_no_macro" not in config  # renombrado en Fase 3
