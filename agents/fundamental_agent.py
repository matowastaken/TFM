"""
agents/fundamental_agent.py

Analista fundamental macro/geopolítico (Fase 2.3 / 3.3). Comparte fuente,
ventana y presupuesto de caracteres con sentiment_agent.py (decisión de
Fase 1.5 / 2.3: news_items está indexado por compañía, no existe un pool
macro independiente en el dataset). La diferenciación entre ambos roles
está en la tarea del prompt, no en el input.
"""

from __future__ import annotations

import logging

import pandas as pd

from agents._common import format_news_block, select_news, validate_analyst_output
from llm.call_llm import call_llm

logger = logging.getLogger(__name__)

ROLE = "fundamental"

SYSTEM_PROMPT = """\
You are a macro/geopolitical analyst at an investment firm. Your report will be read by
the trader, a colleague who will use it, together with other inputs, to place a market order
for this ticker after today's close; it will be filled at the next session's open, which is
not yet known. The trader does not see the raw data. Describe how the broader macroeconomic
and geopolitical context (interest rates, inflation, monetary policy, elections, geopolitical
events, sector-wide or commodity conditions) is likely to affect this ticker over the next few
trading days; do NOT recommend an order yourself. Ignore company-specific news that has no
macro/geopolitical dimension (e.g. product launches, earnings beats with no broader context).
Analyse ONLY the news items provided. Do not use outside knowledge about the company, events
after the given date, or how the market actually reacted. If no news items with macro/geopolitical
relevance are provided, say so and return a neutral signal with low confidence. Respond with a
single JSON object and nothing else."""

USER_TEMPLATE = """\
Ticker: {ticker}
Analysis date: {date}

News items from the last 3 days (date, text):
{news_block}

Task: assess the macroeconomic/geopolitical outlook for {ticker} over the next few trading days,
focusing on what is most relevant for a trader deciding on an order today.
Return JSON with keys:
"role" (always "fundamental"), "signal" ("bullish"|"neutral"|"bearish"),
"score" (float in [-1,1]), "confidence" (float in [0,1]),
"rationale" (max 50 words, referencing the specific news items used)."""


def run(date: pd.Timestamp, ticker: str, news_items: pd.DataFrame) -> dict | None:
    """
    Misma firma y misma fuente de datos que sentiment_agent.run(); ver ahí
    para el detalle de `news_items`. Devuelve None si call_llm falla de
    forma persistente -- el llamador debe tratarlo como un día/ticker
    fallido.
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
        # La respuesta llegó (HTTP 200, JSON válido) pero no cumple el esquema:
        # se descarta igual que antes, pero se deja rastro para el QA (Fase 4.2).
        logger.warning("analista fundamental: respuesta descartada por no validar el esquema: %r", output)
        return None

    return output