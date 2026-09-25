"""
Prototipo del analista técnico (Fase 2.1).

- Indicadores en pandas puro, calculados solo con datos hasta `date` (sin look-ahead).
- Prompt con audiencia (el trader) y ejecución a la apertura siguiente (next_open).
- Llamada a Azure AI Foundry con client.responses.create() y gpt-4.1-mini.
- Validación del JSON y medida de variabilidad entre repeticiones.

Variables de entorno (.env): AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT, FINSABER_DATA_PATH
Uso: python technical_prototype.py --ticker NVDA --repeats 3
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
WARMUP_DAYS = 120  # días naturales previos al inicio del test para calentar SMA50/MACD

SYSTEM_PROMPT = """You are a technical analyst at an investment firm. Your report will be read by \
the trader, a colleague who will use it, together with other inputs, to place a market order for \
this ticker after today's close; it will be filled at the next session's open, which is not yet \
known. The trader does not see the raw data. Describe what the data implies for the next few \
trading days; do NOT recommend an order yourself.
Analyse ONLY the data provided. Do not use outside knowledge about the company, news or events \
after the given date. Respond with a single JSON object and nothing else."""

USER_TEMPLATE = """Ticker: {ticker}
Analysis date: {date}

Last 5 sessions (date, open, high, low, close, volume):
{ohlcv_table}

Indicators as of {date}:
- SMA20: {sma20} | SMA50: {sma50} | Close: {close}
- RSI(14): {rsi}
- MACD: {macd} | Signal: {macd_signal} | Histogram: {macd_hist}
- Bollinger(20,2): lower {bb_low} | mid {bb_mid} | upper {bb_up}
- ATR(14): {atr}
- Relative volume (vs 20d avg): {rel_vol}

Task: assess the technical outlook over the next few trading days, focusing on what is most \
relevant for a trader deciding on an order today.
Return JSON with keys:
"role" (always "technical"), "signal" ("bullish"|"neutral"|"bearish"),
"score" (float in [-1,1]), "confidence" (float in [0,1]),
"rationale" (max 50 words, citing the specific indicators used)."""


# --------------------------------------------------------------------------- datos
def load_prices(ticker: str, data_path: str) -> pd.DataFrame:
    """Lee price_daily (parquet particionado) para un ticker. Usa columnas ajustadas."""
    import pyarrow.dataset as ds

    dataset = ds.dataset(str(Path(data_path) / "price_daily"), format="parquet", partitioning="hive")
    cols = ["date", "symbol", "open", "high", "low", "close", "adjusted_close", "volume"]
    df = dataset.to_table(columns=cols, filter=ds.field("symbol") == ticker).to_pandas()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    # El parquet no trae adjusted_open/high/low: FINSABER los calcula al cargar como
    # precio_bruto * (adjusted_close / close). Se replica aquí para usar los mismos precios.
    adj = (df["adjusted_close"] / df["close"]).where(df["close"] != 0, 0)
    out = pd.DataFrame({
        "open": df["open"] * adj, "high": df["high"] * adj,
        "low": df["low"] * adj, "close": df["adjusted_close"],
        "volume": df["volume"],
    })
    start = pd.Timestamp(TEST_START) - pd.Timedelta(days=WARMUP_DAYS)
    return out.loc[start:TEST_END]


# --------------------------------------------------------------------------- indicadores
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Todos los indicadores son causales: el valor en t solo usa datos <= t."""
    out = df.copy()
    c, h, l, v = out["close"], out["high"], out["low"], out["volume"]

    out["sma20"] = c.rolling(20).mean()
    out["sma50"] = c.rolling(50).mean()

    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    out["rsi"] = 100 - 100 / (1 + gain / loss)

    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    std20 = c.rolling(20).std(ddof=0)
    out["bb_mid"] = out["sma20"]
    out["bb_up"] = out["bb_mid"] + 2 * std20
    out["bb_low"] = out["bb_mid"] - 2 * std20
    out["bb_width"] = (out["bb_up"] - out["bb_low"]) / out["bb_mid"]

    prev_close = c.shift(1)
    tr = pd.concat([h - l, (h - prev_close).abs(), (l - prev_close).abs()], axis=1).max(axis=1)
    out["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()

    out["rel_vol"] = v / v.rolling(20).mean()
    return out


def build_prompt(ticker: str, date, ind: pd.DataFrame) -> str:
    """Construye el prompt de usuario usando SOLO filas <= date."""
    date = pd.Timestamp(date)
    hist = ind.loc[:date]
    row = hist.iloc[-1]
    last5 = hist[["open", "high", "low", "close", "volume"]].tail(5)
    table = "\n".join(
        f"{d.date()}, {r.open:.2f}, {r.high:.2f}, {r.low:.2f}, {r.close:.2f}, {int(r.volume)}"
        for d, r in last5.iterrows()
    )
    f = lambda x: f"{x:.2f}"
    return USER_TEMPLATE.format(
        ticker=ticker, date=hist.index[-1].date(), ohlcv_table=table,
        sma20=f(row.sma20), sma50=f(row.sma50), close=f(row.close), rsi=f(row.rsi),
        macd=f(row.macd), macd_signal=f(row.macd_signal), macd_hist=f(row.macd_hist),
        bb_low=f(row.bb_low), bb_mid=f(row.bb_mid), bb_up=f(row.bb_up),
        atr=f(row.atr), rel_vol=f(row.rel_vol),
    )


def pick_cases(ind: pd.DataFrame) -> dict:
    """Elige 3 fechas contrastadas dentro del periodo de test."""
    test = ind.loc[TEST_START:TEST_END].dropna()
    return {
        "rsi_max (sobrecompra)": test["rsi"].idxmax(),
        "rsi_min (sobreventa)": test["rsi"].idxmin(),
        "lateral (Bollinger mas estrecho)": test["bb_width"].idxmin(),
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


def validate(raw: str) -> tuple[dict | None, list[str]]:
    """Devuelve (json parseado o None, lista de problemas)."""
    problems = []
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return None, ["JSON no parseable"]
    expected = {"role", "signal", "score", "confidence", "rationale"}
    if set(obj) != expected:
        problems.append(f"claves distintas a las esperadas: {sorted(set(obj) ^ expected)}")
        return obj, problems
    if obj["role"] != "technical":
        problems.append("role != technical")
    if obj["signal"] not in {"bullish", "neutral", "bearish"}:
        problems.append("signal fuera de dominio")
    if not isinstance(obj["score"], (int, float)) or not -1 <= obj["score"] <= 1:
        problems.append("score fuera de [-1,1]")
    if not isinstance(obj["confidence"], (int, float)) or not 0 <= obj["confidence"] <= 1:
        problems.append("confidence fuera de [0,1]")
    if isinstance(obj["rationale"], str) and len(obj["rationale"].split()) > 50:
        problems.append("rationale > 50 palabras")
    # coherencia signal <-> score (umbral 0.15, ajustable)
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
    ap.add_argument("--out", default="technical_prototype_log.jsonl")
    args = ap.parse_args()

    client = OpenAI(api_key=os.environ["AZURE_OPENAI_KEY"], base_url=os.environ["AZURE_OPENAI_ENDPOINT"])
    ind = compute_indicators(load_prices(args.ticker, os.environ["FINSABER_DATA_PATH"]))

    with open(args.out, "a", encoding="utf-8") as log:
        for label, date in pick_cases(ind).items():
            prompt = build_prompt(args.ticker, date, ind)
            print(f"\n=== {label} | {args.ticker} | {pd.Timestamp(date).date()} ===")
            signals, scores = [], []
            for i in range(args.repeats):
                raw = call_azure(client, prompt)
                obj, problems = validate(raw)
                print(f"[{i + 1}] problemas: {problems or 'ninguno'}")
                print(f"    {raw.strip()}")
                if obj and not problems:
                    signals.append(obj["signal"])
                    scores.append(obj["score"])
                log.write(json.dumps({"case": label, "ticker": args.ticker, "date": str(pd.Timestamp(date).date()),
                                      "run": i + 1, "raw": raw, "problems": problems}, ensure_ascii=False) + "\n")
            if scores:
                print(f"--> señales: {dict(Counter(signals))} | score medio {np.mean(scores):.2f} "
                      f"(desv. {np.std(scores):.2f})")


if __name__ == "__main__":
    main()
