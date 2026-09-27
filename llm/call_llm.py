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
    fallido, para revisión en el control de calidad de la Fase 4.2. Esto
    evita que un fallo generalizado (p. ej. una caída prolongada del
    endpoint) dispare reintentos en cascada sobre todo el histórico y
    consuma tokens de forma descontrolada.
  - Bug conocido (Fase 0.4): el diccionario OPENAI_PRICE del monitor de
    coste de FINSABER-2 no incluye gpt-4.1-mini. Se parchea al importar
    este módulo, antes de que cualquier llamada pueda registrarse con
    coste 0.0.
"""

import json
import logging
import os
import time

from openai import OpenAI, APITimeoutError, RateLimitError

import backtest.toolkit.llm_cost_monitor as _cost_monitor
from backtest.toolkit.llm_cost_monitor import add_openai_cost_from_response

logger = logging.getLogger(__name__)

# --- Parche obligatorio (hallazgo Fase 0.4, corregido en Fase 3.5) ---
# Precio confirmado (Fase 3.2): Azure AI Foundry, Global Standard.
_cost_monitor.OPENAI_PRICE["gpt-4.1-mini"] = {"input": 0.40, "output": 1.60}

# El hallazgo original de Fase 0.4 ("OPENAI_PRICE no incluye gpt-4.1-mini")
# estaba incompleto: _openai_cost() en el paquete real (backtest.toolkit.
# llm_cost_monitor, instalado vía finsaber==2.0.1) NO busca el modelo en
# OPENAI_PRICE por clave -- comprueba subcadenas literales hardcodeadas
# ("gpt-4o-mini", "gpt-4o"), y "gpt-4.1-mini" no coincide con ninguna.
# Verificado en vivo (Fase 3.5): con solo el parche de OPENAI_PRICE, el
# coste se registraba igualmente como 0.0. Se reemplaza _openai_cost por
# una versión que sí busca por clave exacta en OPENAI_PRICE.
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
            add_openai_cost_from_response(response)
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