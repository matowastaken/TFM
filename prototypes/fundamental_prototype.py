"""
Prototipo del analista fundamental (macro/geopolítico) (Fase 2.3).

Reutiliza la misma fuente y mecánica de datos que el analista de sentimiento
(news_items filtrado por ticker, ventana de 3 días, selección por presupuesto de
caracteres — ver entregable_fase2.md, 2.3: la Opción B de un pool de noticias
macro sin ticker no es viable con este dataset, `symbol`/`cik` son obligatorios
en cada fila de news_items). La diferenciación frente a sentimiento está en la
tarea del prompt: aquí se le pide ignorar lo específico de la compañía sin
dimensión macro/geopolítica. Sin filtros de contenido previos: el analista lee
cualquier noticia de la ventana tal cual llega.

Variables de entorno (.env): AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT, FINSABER_DATA_PATH
Uso: python fundamental_prototype.py --ticker XOM --repeats 3
"""
import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# Reutiliza la carga de noticias y la selección por presupuesto de caracteres
# ya validadas en el prototipo de sentimiento (misma fuente, mismos parámetros).
from sentiment_prototype import (
    CHAR_BUDGET,
    WINDOW_DAYS,
    load_news,
    pick_cases,
    select_news,
)

MODEL = "gpt-4.1-mini"
TEMPERATURE = 0.2
TEST_START, TEST_END = "2024-08-01", "2025-01-31"

SYSTEM_PROMPT = """You are a macro/geopolitical analyst at an investment firm. Your report will \
be read by the trader, a colleague who will use it, together with other inputs, to place a \
market order for this ticker after today's close; it will be filled at the next session's open, \
which is not yet known. The trader does not see the raw data. Describe how the broader \
macroeconomic and geopolitical context (interest rates, inflation, monetary policy, elections, \
geopolitical events, sector-wide or commodity conditions) is likely to affect this ticker over \
the next few trading days; do NOT recommend an order yourself. Ignore company-specific news that \
has no macro/geopolitical dimension (e.g. product launches, earnings beats with no broader \
context).
Analyse ONLY the news items provided. Do not use outside knowledge about the company or events \
after the given date. If no news items with macro/geopolitical relevance are provided, say so and \
return a neutral signal with low confidence. Respond with a single JSON object and nothing else."""

USER_TEMPLATE = """Ticker: {ticker}
Analysis date: {date}

News items from the last {window} days (date, text):
{news_block}

Task: assess the macroeconomic/geopolitical outlook for {ticker} over the next few trading days, \
focusing on what is most relevant for a trader deciding on an order today.
Return JSON with keys:
"role" (always "fundamental"), "signal" ("bullish"|"neutral"|"bearish"),
"score" (float in [-1,1]), "confidence" (float in [0,1]),
"rationale" (max 50 words, referencing the specific news items used)."""

NO_NEWS_BLOCK = "(no news items available in this window)"


def build_prompt(ticker: str, date, news: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    date = pd.Timestamp(date)
    used = select_news(news, date, window_days=WINDOW_DAYS, char_budget=CHAR_BUDGET)
    if used.empty:
        block = NO_NEWS_BLOCK
    else:
        block = "\n".join(f"{r.date.date()}: {r.news_text}" for r in used.itertuples())
    prompt = USER_TEMPLATE.format(ticker=ticker, date=date.date(), window=WINDOW_DAYS, news_block=block)
    return prompt, used


# --------------------------------------------------------------------------- LLM
def call_azure(client, user_prompt: str) -> str:
    response = client.responses.create(
        model=MODEL,
        instructions=SYSTEM_PROMPT,
        input=user_prompt,
        temperature=TEMPERATURE,
        text={"format": {"type": "json_object"}},
    )
    return response.output_text


def validate(raw: str, news_available: bool) -> tuple[dict | None, list[str]]:
    problems = []
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return None, ["JSON no parseable"]
    expected = {"role", "signal", "score", "confidence", "rationale"}
    if set(obj) != expected:
        problems.append(f"claves distintas a las esperadas: {sorted(set(obj) ^ expected)}")
        return obj, problems
    if obj["role"] != "fundamental":
        problems.append("role != fundamental")
    if obj["signal"] not in {"bullish", "neutral", "bearish"}:
        problems.append("signal fuera de dominio")
    if not isinstance(obj["score"], (int, float)) or not -1 <= obj["score"] <= 1:
        problems.append("score fuera de [-1,1]")
    if not isinstance(obj["confidence"], (int, float)) or not 0 <= obj["confidence"] <= 1:
        problems.append("confidence fuera de [0,1]")
    if isinstance(obj["rationale"], str) and len(obj["rationale"].split()) > 50:
        problems.append("rationale > 50 palabras")
    # nota: aquí "sin noticias" solo cubre el caso de la ventana vacía; el caso
    # "hay noticias pero ninguna con dimensión macro" no se puede chequear
    # automáticamente (requeriría juzgar el contenido) y se revisa a mano.
    if not news_available and obj.get("confidence", 1) > 0.4:
        problems.append("confidence alta sin noticias disponibles")
    if not problems:
        s, sc = obj["signal"], obj["score"]
        if (s == "bullish" and sc <= 0.15) or (s == "bearish" and sc >= -0.15) or (s == "neutral" and abs(sc) > 0.4):
            problems.append(f"signal/score incoherentes ({s}, {sc})")
    return obj, problems


def looks_company_specific(rationale: str) -> bool:
    """Heurística simple para detectar si el rationale parece hablar de la compañía
    y no de contexto macro (revisar a mano, esto es solo una señal de aviso)."""
    macro_terms = ["rate", "inflation", "fed", "geopolit", "election", "opec", "tariff",
                   "monetary", "commodity", "oil price", "treasury", "gdp", "recession"]
    return not any(t in rationale.lower() for t in macro_terms)


def main():
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="XOM")  # XOM: perfil macro/geopolítico (Fase 1.2)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="fundamental_prototype_log.jsonl")
    args = ap.parse_args()

    client = OpenAI(api_key=os.environ["AZURE_OPENAI_KEY"], base_url=os.environ["AZURE_OPENAI_ENDPOINT"])
    news = load_news(args.ticker, os.environ["FINSABER_DATA_PATH"])

    with open(args.out, "a", encoding="utf-8") as log:
        for label, date in pick_cases(news).items():
            prompt, used = build_prompt(args.ticker, date, news)
            print(f"\n=== {label} | {args.ticker} | {pd.Timestamp(date).date()} | "
                  f"{len(used)} noticias, {used['news_text'].str.len().sum() if len(used) else 0} chars ===")
            signals, scores, off_topic = [], [], 0
            for i in range(args.repeats):
                raw = call_azure(client, prompt)
                obj, problems = validate(raw, news_available=not used.empty)
                print(f"[{i + 1}] problemas: {problems or 'ninguno'}")
                print(f"    {raw.strip()}")
                if obj and not problems:
                    signals.append(obj["signal"])
                    scores.append(obj["score"])
                    if looks_company_specific(obj.get("rationale", "")):
                        off_topic += 1
                        print("    [aviso] rationale sin terminos macro reconocibles, revisar a mano")
                log.write(json.dumps({"case": label, "ticker": args.ticker, "date": str(pd.Timestamp(date).date()),
                                      "n_news": len(used), "run": i + 1, "raw": raw, "problems": problems},
                                     ensure_ascii=False) + "\n")
            if scores:
                print(f"--> señales: {dict(Counter(signals))} | score medio {np.mean(scores):.2f} "
                      f"(desv. {np.std(scores):.2f}) | posibles fuera de foco: {off_topic}/{len(scores)}")


if __name__ == "__main__":
    main()