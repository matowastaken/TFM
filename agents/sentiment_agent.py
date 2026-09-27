"""
agents/sentiment_agent.py

Analista de sentimiento (Fase 2.2 / 3.3). Usa `news_items` filtrado por
ticker, ventana de 3 días naturales, presupuesto de 6.000 caracteres
(agents/_common.py). Caso sin noticias contemplado explícitamente.
"""

from __future__ import annotations

import pandas as pd

from agents._common import format_news_block, select_news, validate_analyst_output
from llm.call_llm import call_llm

ROLE = "sentiment"

SYSTEM_PROMPT = """\
You are a sentiment analyst at an investment firm. Your report will be read by
the trader, a colleague who will use it, together with other inputs, to place
a market order for this ticker after today's close; it will be filled at the
next session's open, which is not yet known. The trader does not see the raw
data. Describe how recent news coverage is likely to be perceived by the
market over the next few trading days; do NOT recommend an order yourself.
Analyse ONLY the news items provided. Do not use outside knowledge about the
company, events after the given date, or how the market actually reacted.
If no news items are provided, say so and return a neutral signal with low
confidence. Respond with a single JSON object and nothing else."""

USER_TEMPLATE = """\
Ticker: {ticker}
Analysis date: {date}

News items from the last 3 days (date, text):
{news_block}

Task: assess market sentiment toward {ticker} over the next few trading days,
focusing on what is most relevant for a trader deciding on an order today.
Return JSON with keys:
"role" (always "sentiment"), "signal" ("bullish"|"neutral"|"bearish"),
"score" (float in [-1,1]), "confidence" (float in [0,1]),
"rationale" (max 50 words, referencing the specific news items used)."""


def run(date: pd.Timestamp, ticker: str, news_items: pd.DataFrame) -> dict | None:
    """
    date: fecha de análisis.
    ticker: símbolo bursátil.
    news_items: DataFrame ya filtrado por symbol == ticker, columnas
        incluyendo al menos `date` y `news_text` (esquema de
        FinsaberParquetDataset._read_news). No es necesario pre-filtrar
        por ventana: run() aplica la selección de _common.select_news().

    Devuelve el dict de salida del rol, o None si call_llm falla de forma
    persistente -- el llamador debe tratarlo como un día/ticker fallido.
    """
    news = select_news(news_items, date)

    user_prompt = USER_TEMPLATE.format(
        ticker=ticker,
        date=date.date(),
        news_block=format_news_block(news),
    )

    output = call_llm(SYSTEM_PROMPT, user_prompt)
    if output is None:
        return None

    if not validate_analyst_output(output, ROLE):
        return None

    return output
