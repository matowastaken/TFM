"""
agents/_common.py

Utilidades compartidas por los 4 agentes (Fase 3.3). No es un agente en sí
mismo -- de ahí el prefijo "_" -- sino un módulo interno de apoyo:
  - Validación de los dos esquemas de salida (analistas / trader).
  - Selección de noticias por ventana temporal + presupuesto de caracteres,
    reutilizada por sentiment_agent.py y fundamental_agent.py (comparten
    fuente e input, decisión de Fase 1.5 / 2.3).

Vive dentro de agents/ porque solo lo usan los agentes, pero al no exponer
un run() no se lista como un quinto "módulo independiente" del cronograma.
"""

from __future__ import annotations

import pandas as pd

ANALYST_SIGNALS = {"bullish", "neutral", "bearish"}
TRADER_ACTIONS = {"buy", "hold", "sell"}

NEWS_WINDOW_DAYS = 3
NEWS_CHAR_BUDGET = 6000  # provisional, Fase 2.2 -- pendiente de ajustar por ticker


def validate_analyst_output(output: dict, expected_role: str) -> bool:
    """Valida el esquema común a los 3 analistas (técnico/sentimiento/
    fundamental): role, signal, score, confidence, rationale."""
    try:
        if output.get("role") != expected_role:
            return False
        if output.get("signal") not in ANALYST_SIGNALS:
            return False
        score = output.get("score")
        if not isinstance(score, (int, float)) or not (-1.0 <= score <= 1.0):
            return False
        confidence = output.get("confidence")
        if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
            return False
        if not isinstance(output.get("rationale"), str) or not output["rationale"].strip():
            return False
        return True
    except (TypeError, AttributeError):
        return False


def validate_trader_output(output: dict) -> bool:
    """Valida el esquema del trader: action, size, confidence, rationale."""
    try:
        if output.get("action") not in TRADER_ACTIONS:
            return False
        size = output.get("size")
        if not isinstance(size, (int, float)) or not (0.0 <= size <= 1.0):
            return False
        confidence = output.get("confidence")
        if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
            return False
        if not isinstance(output.get("rationale"), str) or not output["rationale"].strip():
            return False
        return True
    except (TypeError, AttributeError):
        return False


def select_news(
    news_items: pd.DataFrame,
    date: pd.Timestamp,
    window_days: int = NEWS_WINDOW_DAYS,
    char_budget: int = NEWS_CHAR_BUDGET,
) -> pd.DataFrame:
    """Selecciona noticias de `news_items` (ya filtrado por symbol == ticker
    por el llamador) dentro de la ventana [date - window_days, date],
    ordenadas de más a menos reciente, hasta que la suma acumulada de
    caracteres de `news_text` alcanza `char_budget`. La noticia que cruza
    el umbral se incluye completa; no se añade ninguna más después.

    Devuelve un DataFrame vacío si no hay noticias en la ventana (el
    agente debe tratar esto como "sin noticias", no como un error).
    """
    window_start = date - pd.Timedelta(days=window_days)
    windowed = news_items[
        (news_items["date"] >= window_start) & (news_items["date"] <= date)
    ].sort_values("date", ascending=False)

    if windowed.empty:
        return windowed

    cum_chars = windowed["news_text"].str.len().cumsum()
    cutoff_idx = cum_chars[cum_chars >= char_budget].index
    if len(cutoff_idx) == 0:
        return windowed  # todas caben dentro del presupuesto
    cutoff_pos = windowed.index.get_loc(cutoff_idx[0])
    return windowed.iloc[: cutoff_pos + 1]


def format_news_block(news: pd.DataFrame) -> str:
    """Da formato de texto plano (date, text) a un DataFrame de noticias
    ya seleccionado, para insertar en el prompt."""
    if news.empty:
        return "(no news items available for this window)"
    lines = [
        f"{row['date'].date()}: {row['news_text']}" for _, row in news.iterrows()
    ]
    return "\n".join(lines)
