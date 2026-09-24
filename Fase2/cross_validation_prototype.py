"""
Validación cruzada de formato — Fase 2.5.

Prueba los 4 prompts ya cerrados (técnico, sentimiento, fundamental, trader) contra Azure
(gpt-4.1-mini, Responses API) y Gemini (gemini-3.5-flash-lite, Interactions API), con el
MISMO input fijo en ambos casos, y compara validez de formato y coherencia cualitativa.

Alcance de esta fase (cronograma 2.5): SOLO contraste de formato/calidad entre proveedores y
documentación de la decisión de asignación (Azure motor único de los 4 roles; Gemini solo para
este contraste y como red de seguridad puntual — Fase 0.2). La lógica de reintentos/fallback
automático NO es parte de esta fase: se implementa en el wrapper call_llm() de la Fase 3.2.

La llamada a Gemini usa los nombres propios de la Interactions API
(system_instruction/input/generation_config), manteniendo la salida JSON para compararla con
Azure.

Usa prompts fijos autocontenidos (no recalculados desde FINSABER-2 ni desde news_items real),
para poder ejecutarse sin dataset ni pipeline de datos, solo con las dos claves de API.

Variables de entorno (.env): AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT, GEMINI_API_KEY
Uso: python cross_validation_prototype.py --repeats 2
"""
import argparse
import json
import os

# Reutiliza los prompts y validadores ya cerrados de cada rol (deben estar en la misma carpeta).
import technical_prototype as technical
import sentiment_prototype as sentiment
import fundamental_prototype as fundamental
import trader_prototype as trader

AZURE_MODEL = "gpt-4.1-mini"
GEMINI_MODEL = "gemini-3.5-flash-lite"
TEMPERATURE = 0.2

# --------------------------------------------------------------------------- casos fijos (sin dataset)
TECHNICAL_SAMPLE_USER = technical.USER_TEMPLATE.format(
    ticker="NVDA", date="2024-10-14",
    ohlcv_table=(
        "2024-10-08, 122.30, 124.10, 121.80, 123.50, 210000000\n"
        "2024-10-09, 123.60, 125.90, 123.00, 125.40, 198000000\n"
        "2024-10-10, 125.50, 128.20, 125.10, 127.80, 225000000\n"
        "2024-10-11, 127.90, 130.50, 127.60, 129.90, 240000000\n"
        "2024-10-14, 130.00, 134.20, 129.80, 133.70, 260000000"
    ),
    sma20="118.40", sma50="109.20", close="133.70", rsi="70.45",
    macd="3.85", macd_signal="2.90", macd_hist="0.95",
    bb_low="108.10", bb_mid="118.40", bb_up="128.70",
    atr="3.20", rel_vol="0.92",
)

SENTIMENT_SAMPLE_USER = sentiment.USER_TEMPLATE.format(
    ticker="NVDA", date="2024-10-14", window=sentiment.WINDOW_DAYS,
    news_block=(
        "2024-10-14: Nvidia shares hit an all-time high as AI chip demand remains strong.\n"
        "2024-10-13: Analysts raise price targets on Nvidia ahead of next earnings report.\n"
        "2024-10-12: Some investors flag valuation concerns after Nvidia's rapid 2024 rally."
    ),
)

FUNDAMENTAL_SAMPLE_USER = fundamental.USER_TEMPLATE.format(
    ticker="XOM", date="2024-10-11", window=fundamental.WINDOW_DAYS,
    news_block=(
        "2024-10-11: Oil prices climb as OPEC+ signals extended production cuts.\n"
        "2024-10-10: Federal Reserve officials hint at a slower pace of rate cuts.\n"
        "2024-10-09: Geopolitical tensions in the Middle East raise supply concerns."
    ),
)

TRADER_SAMPLE_USER = trader.build_prompt(
    "NVDA", "2024-10-14", trader.POSITION_NONE,
    [trader.TECH_BULLISH_NVDA, trader.SENTIMENT_NEUTRAL_NVDA, trader.FUNDAMENTAL_NEUTRAL_NVDA_SYNTH],
)

CASES = [
    ("technical", technical.SYSTEM_PROMPT, TECHNICAL_SAMPLE_USER, technical.validate, {}),
    ("sentiment", sentiment.SYSTEM_PROMPT, SENTIMENT_SAMPLE_USER, sentiment.validate, {"news_available": True}),
    ("fundamental", fundamental.SYSTEM_PROMPT, FUNDAMENTAL_SAMPLE_USER, fundamental.validate, {"news_available": True}),
    ("trader", trader.SYSTEM_PROMPT, TRADER_SAMPLE_USER, trader.validate, {"position_state": trader.POSITION_NONE}),
]


# --------------------------------------------------------------------------- llamadas a proveedores
def call_azure(client, system_prompt: str, user_prompt: str) -> str:
    response = client.responses.create(
        model=AZURE_MODEL,
        instructions=system_prompt,
        input=user_prompt,
        temperature=TEMPERATURE,
        text={"format": {"type": "json_object"}},
    )
    return response.output_text


def call_gemini(client, system_prompt: str, user_prompt: str) -> str:
    """Ejecuta una interacción Gemini con instrucción de sistema y salida JSON."""
    interaction = client.interactions.create(
        model=GEMINI_MODEL,
        system_instruction=system_prompt,
        input=user_prompt,
        generation_config={
            "temperature": TEMPERATURE,
            "response_mime_type": "application/json",
        },
    )
    return interaction.output_text


# --------------------------------------------------------------------------- comparación
def compare_one(role: str, raw_azure: str, raw_gemini: str, validate_fn, validate_kwargs: dict) -> dict:
    obj_a, problems_a = validate_fn(raw_azure, **validate_kwargs)
    obj_g, problems_g = validate_fn(raw_gemini, **validate_kwargs)
    return {
        "role": role,
        "azure": {"raw": raw_azure, "parsed": obj_a, "problems": problems_a},
        "gemini": {"raw": raw_gemini, "parsed": obj_g, "problems": problems_g},
    }


def main():
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--out", default="cross_validation_log.jsonl")
    args = ap.parse_args()

    azure_client = OpenAI(api_key=os.environ["AZURE_OPENAI_KEY"], base_url=os.environ["AZURE_OPENAI_ENDPOINT"])

    gemini_client = None
    try:
        from google import genai
        gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    except Exception as e:
        print(f"[aviso] no se pudo inicializar el cliente de Gemini ({e}); "
              f"se ejecutará solo Azure. Revisa GEMINI_API_KEY y el import de 'google.genai'.")

    with open(args.out, "a", encoding="utf-8") as log:
        for role, system_prompt, user_prompt, validate_fn, validate_kwargs in CASES:
            print(f"\n=== {role} ===")
            for i in range(args.repeats):
                raw_azure = call_azure(azure_client, system_prompt, user_prompt)
                raw_gemini = None
                if gemini_client is not None:
                    try:
                        raw_gemini = call_gemini(gemini_client, system_prompt, user_prompt)
                    except Exception as e:
                        raw_gemini = json.dumps({"_error": str(e)})

                result = compare_one(role, raw_azure, raw_gemini or "", validate_fn, validate_kwargs)
                print(f"[{i + 1}] Azure  -> problemas: {result['azure']['problems'] or 'ninguno'}")
                print(f"    {raw_azure.strip()}")
                if gemini_client is not None:
                    print(f"[{i + 1}] Gemini -> problemas: {result['gemini']['problems'] or 'ninguno'}")
                    print(f"    {(raw_gemini or '').strip()}")
                    # comparación cualitativa simple: ¿coinciden en la etiqueta categórica?
                    key = "signal" if role != "trader" else "action"
                    a_val = result["azure"]["parsed"].get(key) if result["azure"]["parsed"] else None
                    g_val = result["gemini"]["parsed"].get(key) if result["gemini"]["parsed"] else None
                    if a_val is not None and g_val is not None:
                        match = "coinciden" if a_val == g_val else f"DIFIEREN ({a_val} vs {g_val})"
                        print(f"    -> {key}: {match}")
                log.write(json.dumps({"role": role, "run": i + 1, **result}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()