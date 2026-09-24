"""
Prototipo del analista de sentimiento (Fase 2.2).

- Noticias de los últimos 3 días naturales hasta `date`, seleccionadas por presupuesto
  de caracteres acumulado (se incluye completa la noticia que cruza el límite).
- Prompt con audiencia (el trader), ejecución a la apertura siguiente (next_open),
  y manejo explícito del caso sin noticias.
- Llamada a Azure AI Foundry con client.responses.create() y gpt-4.1-mini.
- Validación del JSON y medida de variabilidad entre repeticiones. Sin filtros de
  contenido previos: el analista lee cualquier noticia de la ventana tal cual llega
  (incluidas las que ya narran una reacción de precio del propio día `date`, que no
  es leakage: es información pública contemporánea a la decisión, igual que el
  cierre/volumen que usa el analista técnico).

Variables de entorno (.env): AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT, FINSABER_DATA_PATH
Uso: python sentiment_prototype.py --ticker NVDA --repeats 3
"""
import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

MODEL = "gpt-4.1-mini"
TEMPERATURE = 0.2
TEST_START, TEST_END = "2024-08-01", "2025-01-31"
WINDOW_DAYS = 3          # noticias de los últimos N días naturales hasta `date` inclusive
CHAR_BUDGET = 6000       # provisional: se ajusta tras observar volumen real por ticker

SYSTEM_PROMPT = """You are a sentiment analyst at an investment firm. Your report will be read by \
the trader, a colleague who will use it, together with other inputs, to place a market order for \
this ticker after today's close; it will be filled at the next session's open, which is not yet \
known. The trader does not see the raw data. Describe how recent news coverage is likely to be \
perceived by the market over the next few trading days; do NOT recommend an order yourself.
Analyse ONLY the news items provided. Do not use outside knowledge about the company or events \
after the given date. If no news items are provided, say so and return a neutral signal with low \
confidence. Respond with a single JSON object and nothing else."""

USER_TEMPLATE = """Ticker: {ticker}
Analysis date: {date}

News items from the last {window} days (date, text):
{news_block}

Task: assess market sentiment toward {ticker} over the next few trading days, focusing on what is \
most relevant for a trader deciding on an order today.
Return JSON with keys:
"role" (always "sentiment"), "signal" ("bullish"|"neutral"|"bearish"),
"score" (float in [-1,1]), "confidence" (float in [0,1]),
"rationale" (max 50 words, referencing the specific news items used)."""

NO_NEWS_BLOCK = "(no news items available in this window)"


# --------------------------------------------------------------------------- datos
def load_news(ticker: str, data_path: str) -> pd.DataFrame:
    """Lee news_items (parquet particionado por year, hive) para un ticker."""
    import pyarrow.dataset as ds

    dataset = ds.dataset(str(Path(data_path) / "news_items"), format="parquet", partitioning="hive")
    cols = ["date", "symbol", "item_index", "news_text"]
    df = dataset.to_table(columns=cols, filter=ds.field("symbol") == ticker).to_pandas()
    df["date"] = pd.to_datetime(df["date"])
    # más reciente primero; item_index como desempate estable dentro del mismo día
    return df.sort_values(["date", "item_index"], ascending=[False, True]).reset_index(drop=True)


def select_news(news: pd.DataFrame, date: pd.Timestamp, window_days: int = WINDOW_DAYS,
                 char_budget: int = CHAR_BUDGET) -> pd.DataFrame:
    """Filtra la ventana de `window_days` hasta `date` inclusive (sin look-ahead) y aplica
    el presupuesto de caracteres: se van añadiendo noticias (más recientes primero) hasta
    que el acumulado alcanza char_budget; la que cruza el umbral se incluye completa y no
    se añade ninguna más después."""
    start = date - pd.Timedelta(days=window_days - 1)
    window = news[(news["date"] >= start) & (news["date"] <= date)]
    selected, total = [], 0
    for _, row in window.iterrows():
        selected.append(row)
        total += len(row["news_text"])
        if total >= char_budget:
            break
    return pd.DataFrame(selected) if selected else window.iloc[0:0]


def build_prompt(ticker: str, date, news: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """Construye el prompt de usuario. Devuelve también las noticias usadas (para
    inspeccionar el caso en el log, sin aplicar ningún filtro de contenido)."""
    date = pd.Timestamp(date)
    used = select_news(news, date)
    if used.empty:
        block = NO_NEWS_BLOCK
    else:
        block = "\n".join(f"{r.date.date()}: {r.news_text}" for r in used.itertuples())
    prompt = USER_TEMPLATE.format(ticker=ticker, date=date.date(), window=WINDOW_DAYS, news_block=block)
    return prompt, used


def pick_cases(news: pd.DataFrame) -> dict:
    """Elige 3 fechas dentro del periodo de test: mucha densidad de noticias, poca/ninguna,
    y una intermedia (mediana)."""
    test_dates = pd.date_range(TEST_START, TEST_END, freq="D")
    counts = pd.Series(
        {d: len(select_news(news, d)) for d in test_dates}
    )
    dense = counts.idxmax()
    sparse = counts.idxmin()  # candidato a caso "sin noticias" o casi
    median_val = counts.quantile(0.5, interpolation="nearest")
    mid = counts[counts == median_val].index[len(counts[counts == median_val]) // 2]
    return {
        "denso (mas noticias en ventana)": dense,
        "escaso (menos noticias en ventana)": sparse,
        "intermedio (mediana)": mid,
    }


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
    if obj["role"] != "sentiment":
        problems.append("role != sentiment")
    if obj["signal"] not in {"bullish", "neutral", "bearish"}:
        problems.append("signal fuera de dominio")
    if not isinstance(obj["score"], (int, float)) or not -1 <= obj["score"] <= 1:
        problems.append("score fuera de [-1,1]")
    if not isinstance(obj["confidence"], (int, float)) or not 0 <= obj["confidence"] <= 1:
        problems.append("confidence fuera de [0,1]")
    if isinstance(obj["rationale"], str) and len(obj["rationale"].split()) > 50:
        problems.append("rationale > 50 palabras")
    if not news_available and obj.get("confidence", 1) > 0.4:
        problems.append("confidence alta sin noticias disponibles")
    if not problems:
        s, sc = obj["signal"], obj["score"]
        if (s == "bullish" and sc <= 0.15) or (s == "bearish" and sc >= -0.15) or (s == "neutral" and abs(sc) > 0.4):
            problems.append(f"signal/score incoherentes ({s}, {sc})")
    return obj, problems


def main():
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="NVDA")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="sentiment_prototype_log.jsonl")
    args = ap.parse_args()

    client = OpenAI(api_key=os.environ["AZURE_OPENAI_KEY"], base_url=os.environ["AZURE_OPENAI_ENDPOINT"])
    news = load_news(args.ticker, os.environ["FINSABER_DATA_PATH"])

    with open(args.out, "a", encoding="utf-8") as log:
        for label, date in pick_cases(news).items():
            prompt, used = build_prompt(args.ticker, date, news)
            print(f"\n=== {label} | {args.ticker} | {pd.Timestamp(date).date()} | "
                  f"{len(used)} noticias, {used['news_text'].str.len().sum() if len(used) else 0} chars ===")
            signals, scores = [], []
            for i in range(args.repeats):
                raw = call_azure(client, prompt)
                obj, problems = validate(raw, news_available=not used.empty)
                print(f"[{i + 1}] problemas: {problems or 'ninguno'}")
                print(f"    {raw.strip()}")
                if obj and not problems:
                    signals.append(obj["signal"])
                    scores.append(obj["score"])
                log.write(json.dumps({"case": label, "ticker": args.ticker, "date": str(pd.Timestamp(date).date()),
                                      "n_news": len(used), "run": i + 1, "raw": raw, "problems": problems},
                                     ensure_ascii=False) + "\n")
            if scores:
                print(f"--> señales: {dict(Counter(signals))} | score medio {np.mean(scores):.2f} "
                      f"(desv. {np.std(scores):.2f})")


if __name__ == "__main__":
    main()
