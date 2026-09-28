"""
agents/technical_agent.py

Analista técnico (Fase 2.1 / 3.3). Calcula indicadores causales en pandas
puro (sin `ta`/`pandas-ta`, decisión de Fase 2.1) a partir del histórico
OHLCV hasta `date` inclusive, y llama al LLM con las últimas 5 sesiones +
los indicadores del día `date`.
"""

from __future__ import annotations

import logging

import pandas as pd

from agents._common import validate_analyst_output
from llm.call_llm import call_llm

logger = logging.getLogger(__name__)

ROLE = "technical"

SYSTEM_PROMPT = """\
You are a technical analyst at an investment firm. Your report will be read
by the trader, a colleague who will use it, together with other inputs, to
place a market order for this ticker after today's close; it will be filled
at the next session's open, which is not yet known. The trader does not see
the raw data. Describe what the technical picture suggests about the next
few trading days; do NOT recommend an order yourself. Analyse ONLY the data
provided. Do not use outside knowledge about the company, events after the
given date, or how the market actually reacted. Respond with a single JSON
object and nothing else."""

USER_TEMPLATE = """\
Ticker: {ticker}
Analysis date: {date}

Last 5 sessions (date, open, high, low, close, volume):
{ohlcv_block}

Indicators as of {date}:
{indicators_block}

Task: assess the technical outlook for {ticker} over the next few trading
days, focusing on what is most relevant for a trader deciding on an order
today. Return JSON with keys:
"role" (always "technical"), "signal" ("bullish"|"neutral"|"bearish"),
"score" (float in [-1,1]), "confidence" (float in [0,1]),
"rationale" (max 50 words, referencing the specific indicators used)."""


def _wilder_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    # avg_loss == 0 (solo subidas en la ventana) -> RSI = 100, no NaN.
    rsi = rsi.where(avg_loss != 0, 100.0)
    return rsi


def compute_indicators(ohlcv_history: pd.DataFrame) -> dict:
    """Calcula SMA20/50, RSI(14) Wilder, MACD(12,26,9), Bollinger(20,2),
    ATR(14) y volumen relativo (vs. media 20d) para la última fila de
    `ohlcv_history`. Todo causal: solo usa datos <= la fecha de esa fila.
    `ohlcv_history` debe venir ordenado por fecha ascendente y contener
    suficiente histórico (>= 50 sesiones) para que SMA50/ATR14 no sean NaN.
    """
    close = ohlcv_history["close"]
    high = ohlcv_history["high"]
    low = ohlcv_history["low"]
    volume = ohlcv_history["volume"]

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    rsi14 = _wilder_rsi(close, 14)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()

    bb_mid = sma20
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std

    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr14 = tr.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()

    vol_ma20 = volume.rolling(20).mean()
    rel_volume = volume / vol_ma20

    last = -1
    return {
        "sma20": sma20.iloc[last],
        "sma50": sma50.iloc[last],
        "rsi14": rsi14.iloc[last],
        "macd": macd_line.iloc[last],
        "macd_signal": signal_line.iloc[last],
        "bollinger_upper": bb_upper.iloc[last],
        "bollinger_lower": bb_lower.iloc[last],
        "atr14": atr14.iloc[last],
        "relative_volume_20d": rel_volume.iloc[last],
    }


def _format_ohlcv_block(ohlcv_last5: pd.DataFrame) -> str:
    lines = [
        f"{idx.date()}: O={row.open:.2f} H={row.high:.2f} L={row.low:.2f} "
        f"C={row.close:.2f} V={int(row.volume)}"
        for idx, row in ohlcv_last5.iterrows()
    ]
    return "\n".join(lines)


def _format_indicators_block(indicators: dict) -> str:
    return "\n".join(f"{k}: {v:.4f}" for k, v in indicators.items() if pd.notna(v))


def run(date: pd.Timestamp, ticker: str, ohlcv_history: pd.DataFrame) -> dict | None:
    """
    date: fecha de análisis (la última fila de ohlcv_history debe ser esta fecha).
    ticker: símbolo bursátil.
    ohlcv_history: DataFrame indexado por fecha (ascendente), columnas
        open/high/low/close/volume, con datos <= date (causal), y
        suficiente histórico previo (>= 50 sesiones) para los indicadores.

    Devuelve el dict de salida del rol, o None si call_llm falla de forma
    persistente (ver llm/call_llm.py) -- el llamador debe tratarlo como un
    día/ticker fallido.
    """
    indicators = compute_indicators(ohlcv_history)
    last5 = ohlcv_history.tail(5)

    user_prompt = USER_TEMPLATE.format(
        ticker=ticker,
        date=date.date(),
        ohlcv_block=_format_ohlcv_block(last5),
        indicators_block=_format_indicators_block(indicators),
    )

    output = call_llm(SYSTEM_PROMPT, user_prompt)
    if output is None:
        return None

    if not validate_analyst_output(output, ROLE):
        # La respuesta llegó (HTTP 200, JSON válido) pero no cumple el esquema:
        # se descarta igual que antes, pero se deja rastro para el QA (Fase 4.2).
        logger.warning("analista técnico: respuesta descartada por no validar el esquema: %r", output)
        return None

    return output