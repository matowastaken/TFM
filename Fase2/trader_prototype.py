"""
Prototipo del trader / decisor final (Fase 2.4).

A diferencia de los tres analistas, el trader no llama a FINSABER-2 directamente: agrega
informes ya producidos (de technical_prototype.py, sentiment_prototype.py,
fundamental_prototype.py) y decide accion + tamaño de posicion. Este prototipo usa informes
de ejemplo fijos (algunos reales, tomados de ejecuciones previas; otros sinteticos, marcados
como tal) para poder probar el prompt y el esquema de salida de forma aislada, con distintas
combinaciones de roles activos (1 a 3 analistas) y el caso baseline sin ninguno.

- size en [0,1]: fraccion de cash (buy) o de la posicion actual (sell) — la traduccion a
  quantity la hace el codigo, no el LLM (ver entregable_fase2.md, 2.4).
- quantity=-1 es seguro en FINSABER-2 porque cada backtest es de un solo ticker con capital
  propio (Fase 1.3); aun asi usamos size explicito por decision de diseño (variable, no solo
  0/1), constante entre las 5 configuraciones.
- El trader aplica criterio propio al ponderar informes (no promedio mecanico), con la misma
  restriccion de no usar conocimiento externo que los analistas.

Variables de entorno (.env): AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT
Uso: python trader_prototype.py --repeats 2
"""
import argparse
import json
import os
from collections import Counter

import numpy as np

MODEL = "gpt-4.1-mini"
TEMPERATURE = 0.2

SYSTEM_PROMPT = """You are the trader at an investment firm. You receive reports from your \
colleague analysts and decide the trading action for this ticker after today's close; the order \
will be filled at the next session's open, which is not yet known. Not all reports are always \
available; base your decision only on the ones provided below.
Use your professional judgment to weigh and reconcile the reports — you are not required to \
average their scores mechanically. For example, give more weight to reports that agree with each \
other than to a single outlier; treat a high-confidence report with caution if its rationale seems \
weak; and prefer a smaller position size when reports conflict, even if their average score looks \
strong.
Do not use outside knowledge about the company or events after the given date beyond what is \
stated in the reports below. Respond with a single JSON object and nothing else."""

USER_TEMPLATE = """Ticker: {ticker}
Analysis date: {date}
Current position: {position_state}

Analyst reports:
{reports_block}

Task: decide the trading action for {ticker} today, considering the analyst reports above and the
current position (e.g. "sell" only makes sense if you currently hold shares).
Return JSON with keys:
"action" ("buy"|"hold"|"sell"),
"size" (float in [0,1]: fraction of available cash to invest if buying, or fraction of the
  current position to liquidate if selling; use 0.0 if holding),
"confidence" (float in [0,1]),
"rationale" (max 60 words, referencing which analyst reports weighed most in the decision)."""

NO_REPORTS_BLOCK = "(no analyst reports available — decide based on current position only)"

POSITION_NONE = "no position (all cash)"
POSITION_HOLDING = "holding shares"


def build_reports_block(reports: list[dict]) -> str:
    if not reports:
        return NO_REPORTS_BLOCK
    blocks = []
    for r in reports:
        blocks.append(
            f"[{r['role']}] signal: {r['signal']} | score: {r['score']} | confidence: {r['confidence']}\n"
            f"  {r['rationale']}"
        )
    return "\n\n".join(blocks)


def build_prompt(ticker: str, date: str, position_state: str, reports: list[dict]) -> str:
    return USER_TEMPLATE.format(
        ticker=ticker, date=date, position_state=position_state,
        reports_block=build_reports_block(reports),
    )


# --------------------------------------------------------------------------- informes de ejemplo
# Reales, tomados de las salidas ya validadas de los prototipos de cada analista (Fase 2.1/2.2/2.3).
TECH_BULLISH_NVDA = {"role": "technical", "signal": "bullish", "score": 0.7, "confidence": 0.85,
    "rationale": "Price above SMA20/50, RSI near 70, MACD positive with rising histogram, close "
                 "near upper Bollinger band. Volume slightly below average but ATR indicates "
                 "moderate volatility supporting upward momentum."}
SENTIMENT_NEUTRAL_NVDA = {"role": "sentiment", "signal": "neutral", "score": 0.1, "confidence": 0.75,
    "rationale": "Mixed signals: Nvidia shares surged 25% recently and a record results "
                 "prediction is positive, but concerns about a potential AI bubble and historical "
                 "post-split price drops temper enthusiasm."}
SENTIMENT_BULLISH_NVDA = {"role": "sentiment", "signal": "bullish", "score": 0.7, "confidence": 0.85,
    "rationale": "Strong AI sector momentum with Nvidia's $700M investment in Nebius, upcoming "
                 "Blackwell chip launch, and partnerships with BYD and OpenAI. Positive market "
                 "sentiment supports a bullish outlook."}
FUNDAMENTAL_NEUTRAL_XOM = {"role": "fundamental", "signal": "neutral", "score": 0.0, "confidence": 0.4,
    "rationale": "Recent news includes ONGC Videsh's gas deal in Azerbaijan, indicating stable "
                 "energy sector activity, but no direct macro or geopolitical shifts impacting "
                 "Exxon Mobil specifically."}

# Sinteticos (no proceden de una ejecucion real del prototipo correspondiente): solo para
# completar combinaciones y probar el prompt/esquema del trader de forma aislada. Marcados
# explicitamente para no confundirlos con datos reales al revisar el log.
SENTIMENT_BEARISH_NVDA_SYNTH = {"role": "sentiment", "signal": "bearish", "score": -0.6, "confidence": 0.7,
    "rationale": "[SYNTHETIC] Reports of a major cloud customer cutting orders and increased "
                 "export restriction concerns weigh on near-term sentiment."}
FUNDAMENTAL_NEUTRAL_NVDA_SYNTH = {"role": "fundamental", "signal": "neutral", "score": 0.05, "confidence": 0.5,
    "rationale": "[SYNTHETIC] No major interest rate or geopolitical shifts noted in the window; "
                 "broader semiconductor sector conditions remain stable."}

# Escenarios: (label, ticker, date, position_state, reports)
SCENARIOS = [
    ("full_system (consenso alcista)", "NVDA", "2024-10-14", POSITION_NONE,
     [TECH_BULLISH_NVDA, SENTIMENT_NEUTRAL_NVDA, FUNDAMENTAL_NEUTRAL_NVDA_SYNTH]),
    ("full_system (señales en conflicto)", "NVDA", "2024-10-14", POSITION_NONE,
     [TECH_BULLISH_NVDA, SENTIMENT_BEARISH_NVDA_SYNTH, FUNDAMENTAL_NEUTRAL_NVDA_SYNTH]),
    ("no_technical (2 roles, holding)", "NVDA", "2024-12-02", POSITION_HOLDING,
     [SENTIMENT_BULLISH_NVDA, FUNDAMENTAL_NEUTRAL_NVDA_SYNTH]),
    ("no_sentiment (2 roles, sin posicion)", "NVDA", "2024-10-14", POSITION_NONE,
     [TECH_BULLISH_NVDA, FUNDAMENTAL_NEUTRAL_NVDA_SYNTH]),
    ("no_fundamental (2 roles, holding)", "NVDA", "2024-10-14", POSITION_HOLDING,
     [TECH_BULLISH_NVDA, SENTIMENT_NEUTRAL_NVDA]),
    ("baseline (sin informes, sin posicion)", "XOM", "2024-09-22", POSITION_NONE, []),
    ("baseline (sin informes, holding)", "XOM", "2024-09-22", POSITION_HOLDING, []),
]


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


def validate(raw: str, position_state: str) -> tuple[dict | None, list[str]]:
    problems = []
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return None, ["JSON no parseable"]
    expected = {"action", "size", "confidence", "rationale"}
    if set(obj) != expected:
        problems.append(f"claves distintas a las esperadas: {sorted(set(obj) ^ expected)}")
        return obj, problems
    if obj["action"] not in {"buy", "hold", "sell"}:
        problems.append("action fuera de dominio")
    if not isinstance(obj["size"], (int, float)) or not 0 <= obj["size"] <= 1:
        problems.append("size fuera de [0,1]")
    if not isinstance(obj["confidence"], (int, float)) or not 0 <= obj["confidence"] <= 1:
        problems.append("confidence fuera de [0,1]")
    if isinstance(obj["rationale"], str) and len(obj["rationale"].split()) > 60:
        problems.append("rationale > 60 palabras")
    if not problems:
        if obj["action"] == "sell" and position_state == POSITION_NONE:
            problems.append("sell sin posicion abierta (inconsistente con current position)")
        if obj["action"] == "hold" and obj["size"] != 0:
            problems.append("hold con size != 0 (no es un error de formato, pero conviene revisar)")
        if obj["action"] in {"buy", "sell"} and obj["size"] == 0:
            problems.append("buy/sell con size 0 (equivale a hold, revisar coherencia)")
    return obj, problems


def main():
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--out", default="trader_prototype_log.jsonl")
    args = ap.parse_args()

    client = OpenAI(api_key=os.environ["AZURE_OPENAI_KEY"], base_url=os.environ["AZURE_OPENAI_ENDPOINT"])

    with open(args.out, "a", encoding="utf-8") as log:
        for label, ticker, date, position_state, reports in SCENARIOS:
            prompt = build_prompt(ticker, date, position_state, reports)
            print(f"\n=== {label} | {ticker} | {date} | posicion: {position_state} | "
                  f"{len(reports)} informe(s) ===")
            actions, sizes = [], []
            for i in range(args.repeats):
                raw = call_azure(client, prompt)
                obj, problems = validate(raw, position_state)
                print(f"[{i + 1}] problemas: {problems or 'ninguno'}")
                print(f"    {raw.strip()}")
                if obj and not problems:
                    actions.append(obj["action"])
                    sizes.append(obj["size"])
                log.write(json.dumps({"case": label, "ticker": ticker, "date": date,
                                      "position_state": position_state, "n_reports": len(reports),
                                      "run": i + 1, "raw": raw, "problems": problems},
                                     ensure_ascii=False) + "\n")
            if actions:
                print(f"--> acciones: {dict(Counter(actions))} | size medio {np.mean(sizes):.2f} "
                      f"(desv. {np.std(sizes):.2f})")


if __name__ == "__main__":
    main()
