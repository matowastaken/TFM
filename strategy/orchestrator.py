"""
strategy/orchestrator.py

Orquestador de configuración (Fase 3.4). Dado un conjunto de roles activos
(según la configuración de ablación), ejecuta únicamente esos agentes
analistas y pasa sus outputs al trader.

Decisión clave (confirmada en Fase 3.4): si un analista activo falla
(agente devuelve None -- ver Fase 3.2/3.3), todo el día/ticker se marca
como fallido en esa configuración. El trader NO se llama con un
subconjunto parcial de informes: hacerlo contaminaría la comparación de
ablación, ya que un día con un fallo técnico del analista técnico se
volvería indistinguible, a nivel de input del trader, de un día normal en
`cherry_pick_no_technical`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from agents import fundamental_agent, sentiment_agent, technical_agent, trader_agent

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "ablation.yaml"

# Roles que consumen news_items (comparten fuente, Fase 1.5/2.3) frente al
# único rol que consume OHLCV.
NEWS_BASED_ROLES = {"sentiment", "fundamental"}

ROLE_AGENTS = {
    "technical": technical_agent,
    "sentiment": sentiment_agent,
    "fundamental": fundamental_agent,
}


@dataclass
class OrchestratorResult:
    setup_name: str
    status: str  # "ok" | "failed"
    failed_role: str | None  # nombre del rol que falló, o "trader"; None si status == "ok"
    analyst_reports: list[dict] | None  # informes obtenidos hasta el punto de fallo (o completos si "ok")
    trader_output: dict | None  # None si status == "failed"


def load_ablation_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, list[str]]:
    """Carga config/ablation.yaml -> {setup_name: [roles activos]}."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return {c["setup_name"]: c["active_roles"] for c in raw["configurations"]}


def run_configuration(
    setup_name: str,
    active_roles: list[str],
    date: pd.Timestamp,
    ticker: str,
    ohlcv_history: pd.DataFrame,
    news_items: pd.DataFrame,
    position_state: str,
) -> OrchestratorResult:
    """
    Ejecuta los agentes analistas de `active_roles` (0 a 3, en el orden
    dado) para (date, ticker), y si todos responden, llama al trader.

    ohlcv_history: histórico OHLCV causal hasta `date` inclusive (ver
        technical_agent.run() para el detalle del formato).
    news_items: noticias ya filtradas por symbol == ticker (ver
        sentiment_agent.run() / fundamental_agent.run()).
    position_state: "no position (all cash)" | "holding shares (fully invested)".

    Devuelve un OrchestratorResult. status == "failed" indica que ese
    día/ticker debe registrarse como fallido para esta configuración
    (Fase 4.2), sin decisión de trading.
    """
    reports: list[dict] = []

    for role in active_roles:
        agent = ROLE_AGENTS[role]
        if role in NEWS_BASED_ROLES:
            output = agent.run(date, ticker, news_items)
        else:
            output = agent.run(date, ticker, ohlcv_history)

        if output is None:
            return OrchestratorResult(
                setup_name=setup_name,
                status="failed",
                failed_role=role,
                analyst_reports=reports or None,
                trader_output=None,
            )
        reports.append(output)

    trader_output = trader_agent.run(date, ticker, position_state, reports)
    if trader_output is None:
        return OrchestratorResult(
            setup_name=setup_name,
            status="failed",
            failed_role="trader",
            analyst_reports=reports,
            trader_output=None,
        )

    return OrchestratorResult(
        setup_name=setup_name,
        status="ok",
        failed_role=None,
        analyst_reports=reports,
        trader_output=trader_output,
    )
