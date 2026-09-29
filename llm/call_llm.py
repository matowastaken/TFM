"""
llm/call_llm.py

Wrapper de llamada al LLM para los 4 roles del sistema multiagente (Fase 3.2).

Decisiones de diseño (ver entregable_fase0.md, entregable_fase2.md y
areas/fase3-implementacion.md):
  - Motor único: Azure AI Foundry, modelo gpt-4.1-mini, vía openai.OpenAI
    (NO AzureOpenAI) y client.responses.create() (Responses API).
  - Gemini NO se usa como fallback dinámico ante excepciones: es solo una
    alternativa MANUAL si se agota el presupuesto/cuota de Azure, para no
    introducir una variable de confusión entre configuraciones de ablación
    ni mezclar proveedores dentro de una misma ejecución del experimento.
  - Ante fallo persistente de Azure (tras MAX_RETRIES intentos), call_llm()
    NO lanza una excepción que detenga todo el backtest: loguea el error y
    devuelve None. El día/ticker correspondiente queda marcado como
    fallido (Fase 4.1: la estrategia lo convierte en DayFailure y el
    runner reintenta con replay).
  - Bug conocido (Fase 0.4): el diccionario OPENAI_PRICE del monitor de
    coste de FINSABER-2 no incluye gpt-4.1-mini. Se parchea al importar
    este módulo, antes de que cualquier llamada pueda registrarse con
    coste 0.0.

Fase 4.1 -- log de llamadas:
  configure_call_log(path) activa un JSONL con una línea por respuesta
  recibida de Azure (incluidas las que luego fallan al parsear JSON, que
  también cuestan tokens): tokens, coste y el contexto fijado por la
  estrategia con set_call_context() (setup, ticker, fecha, intento). El
  coste total del experimento se suma desde este fichero, no desde el
  contador en memoria, que se pierde al reiniciar un backtest. El campo
  system_head (primeros caracteres del prompt de sistema) permite
  distinguir el rol.
"""

import json
import logging
import os
import time
from pathlib import Path

from openai import OpenAI, APITimeoutError, RateLimitError

import backtest.toolkit.llm_cost_monitor as _cost_monitor
from backtest.toolkit.llm_cost_monitor import add_llm_cost

logger = logging.getLogger(__name__)

# --- Parche obligatorio (hallazgo Fase 0.4, corregido en Fase 3.5) ---
# Precio confirmado (Fase 3.2): Azure AI Foundry, Global Standard.
_cost_monitor.OPENAI_PRICE["gpt-4.1-mini"] = {"input": 0.40, "output": 1.60}

# _openai_cost() del paquete real (finsaber==2.0.1) comprueba subcadenas
# literales ("gpt-4o-mini", "gpt-4o") en vez de buscar por clave en
# OPENAI_PRICE, y "gpt-4.1-mini" no coincide con ninguna: con solo el
# parche de OPENAI_PRICE el coste seguía siendo 0.0 (verificado en vivo,
# Fase 3.5). Se reemplaza por una versión que busca por clave exacta.
def _patched_openai_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    price = _cost_monitor.OPENAI_PRICE.get(model)
    if price is None:
        return 0.0
    return (prompt_tokens * price["input"] + completion_tokens * price["output"]) / 1_000_000


_cost_monitor._openai_cost = _patched_openai_cost

AZURE_MODEL = "gpt-4.1-mini"
MAX_RETRIES = 3
BASE_BACKOFF_SECONDS = 1  # 1s, 2s, 4s

_client = OpenAI(
    base_url=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_KEY"],
)

# --- Log de llamadas (Fase 4.1) ---
_call_log_path: Path | None = None
_call_context: dict = {}


def configure_call_log(path) -> None:
    """Activa el log JSONL de llamadas (una línea por respuesta de Azure)."""
    global _call_log_path
    _call_log_path = Path(path)
    _call_log_path.parent.mkdir(parents=True, exist_ok=True)


def set_call_context(**context) -> None:
    """Fija el contexto (setup, ticker, date, attempt) que se añade a cada línea del log."""
    _call_context.clear()
    _call_context.update(context)


def _usage_tokens(response) -> tuple[int, int]:
    """(input_tokens, output_tokens) de una respuesta de la Responses API."""
    usage = getattr(response, "usage", None)
    return (
        getattr(usage, "input_tokens", 0) or 0,
        getattr(usage, "output_tokens", 0) or 0,
    )


def _record_cost(response) -> None:
    """
    Registra el coste en el monitor de FINSABER (metrics.json).

    add_openai_cost_from_response() NO sirve con la Responses API: lee
    usage.prompt_tokens (nombre de la Chat Completions API), que aquí no
    existe (es input_tokens), así que el coste de entrada se contaba como
    0 y solo se registraba el de salida (verificado en Fase 4.1 contra
    calls.jsonl: 4.4x de infraestimación). Se llama a add_llm_cost() con
    los tokens correctos y el modelo fijado a AZURE_MODEL, para que el
    precio se busque por clave exacta y no dependa del nombre versionado
    que devuelva la API.
    """
    prompt_tokens, completion_tokens = _usage_tokens(response)
    add_llm_cost(
        model=AZURE_MODEL,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        provider="openai",
    )


def _log_call(response, system: str) -> None:
    if _call_log_path is None:
        return
    try:
        prompt_tokens, completion_tokens = _usage_tokens(response)
        record = {
            **_call_context,
            "system_head": system[:60],
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost": _patched_openai_cost(AZURE_MODEL, prompt_tokens, completion_tokens),
        }
        with open(_call_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
    except Exception as e:  # el log nunca debe romper una llamada
        logger.warning("call_llm: no se pudo escribir el log de llamadas (%s)", e)


def strip_markdown_json_fence(text: str) -> str:
    """Quita el fence de Markdown (```json ... ```) si el modelo lo añade
    pese a que se le pide JSON puro."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return text.strip()


def call_llm(system: str, user: str, temperature: float = 0.2) -> dict | None:
    """Llama a gpt-4.1-mini vía Azure AI Foundry, con reintentos y backoff
    exponencial ante fallos transitorios (rate limit, timeout, JSON mal
    formado).

    No hay fallback automático a otro proveedor: si Azure falla de forma
    persistente tras MAX_RETRIES intentos, se loguea el error y se
    devuelve None en vez de propagar la excepción. El llamador (agente /
    orquestador) es responsable de tratar un resultado None como un
    día/ticker fallido, sin detener el resto de la ejecución.

    Devuelve el dict ya parseado (esquema de salida del rol) o None.
    """
    last_error: Exception | None = None

    for attempt in range(MAX_RETRIES):
        try:
            response = _client.responses.create(
                model=AZURE_MODEL,
                input=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
            )
            _record_cost(response)
            _log_call(response, system)
            return json.loads(strip_markdown_json_fence(response.output_text))

        except (RateLimitError, APITimeoutError, json.JSONDecodeError) as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                wait = BASE_BACKOFF_SECONDS * (2 ** attempt)
                logger.warning(
                    "call_llm: intento %d/%d fallido (%s: %s), "
                    "reintentando en %ds",
                    attempt + 1, MAX_RETRIES, type(e).__name__, e, wait,
                )
                time.sleep(wait)

    logger.error(
        "call_llm: agotados %d intentos contra Azure (%s: %s). "
        "Se devuelve None; el día/ticker debe marcarse como fallido.",
        MAX_RETRIES, type(last_error).__name__, last_error,
    )
    return None