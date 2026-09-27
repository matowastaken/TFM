"""
tests/test_agents.py

Tests unitarios mínimos de los 4 agentes (Fase 3.3): 2-3 casos por agente,
centrados en el parsing robusto del output, no en la calidad del contenido
(eso ya se validó manualmente en los prototipos de Fase 2). call_llm() se
mockea en todos los casos para no consumir cuota real de Azure.

Ejecutar con: pytest tests/test_agents.py -v
"""

import pandas as pd
import pytest

from agents import fundamental_agent, sentiment_agent, technical_agent, trader_agent


# ---------------------------------------------------------------------------
# Fixtures compartidas
# ---------------------------------------------------------------------------

@pytest.fixture
def ohlcv_history():
    """60 sesiones sintéticas (mínimo para SMA50/ATR14 sin NaN)."""
    dates = pd.date_range("2024-08-01", periods=60, freq="B")
    close = pd.Series(range(100, 160), index=dates, dtype=float)
    return pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": [1_000_000] * 60,
        },
        index=dates,
    )


@pytest.fixture
def news_items():
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-08-05", "2024-08-04", "2024-07-01"]),
            "symbol": ["NVDA", "NVDA", "NVDA"],
            "news_text": [
                "NVDA rallies on strong earnings guidance.",
                "Analysts raise price target for NVDA.",
                "Old news outside the 3-day window.",
            ],
        }
    )


def _valid_analyst_response(role: str):
    return {
        "role": role,
        "signal": "bullish",
        "score": 0.6,
        "confidence": 0.8,
        "rationale": "Strong momentum observed.",
    }


def _valid_trader_response():
    return {
        "action": "buy",
        "size": 0.5,
        "confidence": 0.7,
        "rationale": "Analyst reports agree on upside.",
    }


# ---------------------------------------------------------------------------
# technical_agent
# ---------------------------------------------------------------------------

def test_technical_agent_valid_response(monkeypatch, ohlcv_history):
    monkeypatch.setattr(
        technical_agent, "call_llm", lambda *a, **kw: _valid_analyst_response("technical")
    )
    result = technical_agent.run(ohlcv_history.index[-1], "NVDA", ohlcv_history)
    assert result is not None
    assert result["role"] == "technical"


def test_technical_agent_llm_failure_returns_none(monkeypatch, ohlcv_history):
    monkeypatch.setattr(technical_agent, "call_llm", lambda *a, **kw: None)
    result = technical_agent.run(ohlcv_history.index[-1], "NVDA", ohlcv_history)
    assert result is None


def test_technical_agent_malformed_response_returns_none(monkeypatch, ohlcv_history):
    monkeypatch.setattr(
        technical_agent, "call_llm", lambda *a, **kw: {"role": "technical", "signal": "up"}
    )
    result = technical_agent.run(ohlcv_history.index[-1], "NVDA", ohlcv_history)
    assert result is None


def test_technical_agent_indicators_no_nan_with_enough_history(ohlcv_history):
    indicators = technical_agent.compute_indicators(ohlcv_history)
    assert all(pd.notna(v) for v in indicators.values())


# ---------------------------------------------------------------------------
# sentiment_agent
# ---------------------------------------------------------------------------

def test_sentiment_agent_valid_response(monkeypatch, news_items):
    monkeypatch.setattr(
        sentiment_agent, "call_llm", lambda *a, **kw: _valid_analyst_response("sentiment")
    )
    result = sentiment_agent.run(pd.Timestamp("2024-08-05"), "NVDA", news_items)
    assert result is not None
    assert result["role"] == "sentiment"


def test_sentiment_agent_no_news_in_window_still_calls_llm(monkeypatch, news_items):
    captured = {}

    def fake_call_llm(system, user, **kw):
        captured["user"] = user
        return _valid_analyst_response("sentiment")

    monkeypatch.setattr(sentiment_agent, "call_llm", fake_call_llm)
    # Fecha muy anterior: ninguna noticia cae en la ventana de 3 días.
    result = sentiment_agent.run(pd.Timestamp("2024-01-15"), "NVDA", news_items)
    assert result is not None
    assert "no news items available" in captured["user"]


def test_sentiment_agent_llm_failure_returns_none(monkeypatch, news_items):
    monkeypatch.setattr(sentiment_agent, "call_llm", lambda *a, **kw: None)
    result = sentiment_agent.run(pd.Timestamp("2024-08-05"), "NVDA", news_items)
    assert result is None


# ---------------------------------------------------------------------------
# fundamental_agent
# ---------------------------------------------------------------------------

def test_fundamental_agent_valid_response(monkeypatch, news_items):
    monkeypatch.setattr(
        fundamental_agent, "call_llm", lambda *a, **kw: _valid_analyst_response("fundamental")
    )
    result = fundamental_agent.run(pd.Timestamp("2024-08-05"), "NVDA", news_items)
    assert result is not None
    assert result["role"] == "fundamental"


def test_fundamental_agent_wrong_role_in_response_returns_none(monkeypatch, news_items):
    # Simula que el LLM devuelve "sentiment" en vez de "fundamental" (mismo
    # esquema, error de rol) -- debe fallar la validación.
    monkeypatch.setattr(
        fundamental_agent, "call_llm", lambda *a, **kw: _valid_analyst_response("sentiment")
    )
    result = fundamental_agent.run(pd.Timestamp("2024-08-05"), "NVDA", news_items)
    assert result is None


# ---------------------------------------------------------------------------
# trader_agent
# ---------------------------------------------------------------------------

def test_trader_agent_with_reports(monkeypatch):
    monkeypatch.setattr(trader_agent, "call_llm", lambda *a, **kw: _valid_trader_response())
    reports = [_valid_analyst_response("technical"), _valid_analyst_response("sentiment")]
    result = trader_agent.run(
        pd.Timestamp("2024-08-05"), "NVDA", "no position (all cash)", reports
    )
    assert result is not None
    assert result["action"] == "buy"


def test_trader_agent_baseline_no_reports(monkeypatch):
    captured = {}

    def fake_call_llm(system, user, **kw):
        captured["user"] = user
        return _valid_trader_response()

    monkeypatch.setattr(trader_agent, "call_llm", fake_call_llm)
    result = trader_agent.run(
        pd.Timestamp("2024-08-05"), "NVDA", "no position (all cash)", []
    )
    assert result is not None
    assert trader_agent.NO_REPORTS_PLACEHOLDER in captured["user"]


def test_trader_agent_sell_without_position_still_reaches_llm(monkeypatch):
    # _resolve_quantity resuelve esto de forma segura en FINSABER-2 (Fase 2.4);
    # el agente no debe bloquear la llamada por sí mismo.
    monkeypatch.setattr(
        trader_agent,
        "call_llm",
        lambda *a, **kw: {"action": "sell", "size": 1.0, "confidence": 0.5, "rationale": "x"},
    )
    result = trader_agent.run(
        pd.Timestamp("2024-08-05"), "NVDA", "no position (all cash)", []
    )
    assert result is not None
    assert result["action"] == "sell"
