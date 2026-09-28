"""
agents/trader_agent.py

Trader / decisor final (Fase 2.4 / 3.3). Agrega los informes de los
analistas activos (0 a 3, según la configuración de ablación) con criterio
propio, no mecánico (decisión de Fase 2.4). La traducción `size ->
quantity` NO vive aquí: se hace en código Python puro en la capa de
estrategia (Fase 3.5), para no pedirle al LLM que calcule cash o número de
acciones.
"""

from __future__ import annotations

import logging

import pandas as pd

from agents._common import validate_trader_output
from llm.call_llm import call_llm

logger = logging.getLogger(__name__)

NO_REPORTS_PLACEHOLDER = "(no analyst reports available — decide based on current position only)"

SYSTEM_PROMPT = """\
You are the trader at an investment firm. You receive reports from your colleague analysts and
decide the trading action for this ticker after today's close; the order will be filled at the
next session's open, which is not yet known. Not all reports are always available; base your
decision only on the ones provided below.
Use your professional judgment to weigh and reconcile the reports — you are not required to
average their scores mechanically. For example, give more weight to reports that agree with each
other than to a single outlier; treat a high-confidence report with caution if its rationale seems
weak; and prefer a smaller position size when reports conflict, even if their average score looks
strong.
Do not use outside knowledge about the company or events after the given date beyond what is
stated in the reports below. Respond with a single JSON object and nothing else."""

USER_TEMPLATE = """\
Ticker: {ticker}
Analysis date: {date}
Current position: {position_state}

Analyst reports:
{reports_block}

Task: decide the trading action for {ticker} today, considering the analyst reports above and the
current position (e.g. "sell" only makes sense if you currently hold shares).
Return JSON with keys:
"action" ("buy"|"hold"|"sell"),
"size" (float in [0,1]: fraction of available cash to invest if buying, or fraction of the
  current position to liquidate if selling; ignored if holding),
"confidence" (float in [0,1]),
"rationale" (max 60 words, referencing which analyst reports weighed most in the decision)."""


def _format_reports_block(analyst_reports: list[dict]) -> str:
    if not analyst_reports:
        return NO_REPORTS_PLACEHOLDER
    lines = []
    for report in analyst_reports:
        lines.append(
            f"[{report['role']}] signal={report['signal']} score={report['score']:.2f} "
            f"confidence={report['confidence']:.2f} — {report['rationale']}"
        )
    return "\n".join(lines)


def run(
    date: pd.Timestamp,
    ticker: str,
    position_state: str,
    analyst_reports: list[dict],
) -> dict | None:
    """
    date: fecha de análisis.
    ticker: símbolo bursátil.
    position_state: "no position (all cash)" | "holding shares (fully invested)"
        (leído de framework.cash/framework.portfolio por el llamador, no
        inventado por el LLM -- decisión de Fase 2.4).
    analyst_reports: lista de 0 a 3 dicts, cada uno con el esquema de
        salida de un analista (role/signal/score/confidence/rationale),
        correspondientes a los roles activos en la configuración de
        ablación actual. Lista vacía -> caso baseline (trader-only).

    Devuelve el dict {action, size, confidence, rationale}, o None si
    call_llm falla de forma persistente -- el llamador debe tratarlo como
    un día/ticker fallido.
    """
    user_prompt = USER_TEMPLATE.format(
        ticker=ticker,
        date=date.date(),
        position_state=position_state,
        reports_block=_format_reports_block(analyst_reports),
    )

    output = call_llm(SYSTEM_PROMPT, user_prompt)
    if output is None:
        return None

    # El prompt dice que `size` se ignora si la acción es hold, y el modelo a
    # veces omite la clave en ese caso (visto en el smoke test de la Fase 3.6:
    # 3 de 6 días descartados por esto). Un hold sin `size` es una respuesta
    # válida: se normaliza a 0.0 antes de validar. Para buy/sell `size` sigue
    # siendo obligatorio.
    if isinstance(output, dict) and output.get("action") == "hold" and output.get("size") is None:
        output["size"] = 0.0

    if not validate_trader_output(output):
        # La respuesta llegó (HTTP 200, JSON válido) pero no cumple el esquema:
        # se descarta igual que antes, pero se deja rastro para el QA (Fase 4.2).
        logger.warning("trader: respuesta descartada por no validar el esquema: %r", output)
        return None

    return output