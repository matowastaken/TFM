"""
strategy/ablation_strategy.py

Interfaz de estrategia requerida por FINSABER-2 (Fase 3.5). Hereda de
BaseStrategyIso e implementa on_data(), conectando el orquestador de
configuración (Fase 3.4) con el motor de backtesting.

Puntos clave de la integración real con FINSABER-2 (confirmados en el
código fuente instalado, no solo en la documentación -- ver
areas/fase3-implementacion.md):

  - on_data(date, today_data, framework) se llama UNA vez por fecha, no
    por ticker. today_data solo trae los datos de ese día concreto
    (today_data["price"][ticker], today_data["news"][ticker] = lista de
    textos de ese día), sin histórico.
  - Como ni el analista técnico (>=50 sesiones para SMA50/ATR14) ni
    sentimiento/fundamental (ventana de 3 días, Fase 1.5) pueden trabajar
    solo con el día actual, esta clase mantiene su propio buffer
    acumulado de OHLCV y de noticias, construido incrementalmente en cada
    on_data().
  - Decisión de diseño (Fase 3.5): cada backtest de la Fase 4 instancia
    AblationStrategy una vez por ticker (coherente con el diseño de 1.3:
    20 backtests = 5 configuraciones x 4 tickers, capital independiente
    por ticker). Aunque today_data pueda venir organizado por varios
    tickers, esta estrategia solo lee self.ticker.
  - Warmup (rediseñado en Fase 3.5 a partir de feedback): en vez de dejar
    al rol técnico sin funcionar durante los primeros ~50 días reales de
    cada backtest, el runner (Fase 3.6) debe cargar el dataset ~50
    sesiones ANTES de la fecha real de evaluación (vía
    load_backtest_data_single_ticker con un start_date anterior). Esta
    clase recibe explícitamente evaluation_start_date: acumula OHLCV/
    noticias desde el primer día cargado, pero NO llama al orquestador ni
    opera hasta date >= evaluation_start_date. Así, el primer día real de
    evaluación ya tiene indicadores técnicos completos, sin operar
    "a ciegas" durante el preload ni contaminar el periodo de test con
    decisiones tomadas fuera de la ventana experimental (Fase 1.2).
  - Precios: today_data["price"][ticker] trae campos crudos (open/high/
    low/close) y ajustados por splits/dividendos (adjusted_open/adjusted_
    high/adjusted_low/adjusted_close). La ejecución real de órdenes usa
    los ajustados (_execute_order usa price_field="adjusted_open" para
    fills a next-open, confirmado en backtest_framework_iso.py). Para
    consistencia, los indicadores técnicos y el precio de referencia
    pasado a framework.buy()/.sell() usan también los campos ajustados.
  - Bug de coste (Fase 0.4) corregido de verdad en Fase 3.5: ver
    llm/call_llm.py -- el parche de OPENAI_PRICE por sí solo no bastaba.
"""

from __future__ import annotations

import logging

import pandas as pd
from finsaber.strategy.timing_llm import BaseStrategyIso

from strategy.orchestrator import run_configuration

logger = logging.getLogger(__name__)

MIN_OHLCV_HISTORY_FOR_TECHNICAL = 50  # SMA50/ATR14 (Fase 2.1)


class AblationStrategy(BaseStrategyIso):
    """
    Una instancia por (setup_name, ticker). setup_name identifica la
    configuración de ablación (Fase 1.3/3.4, con el prefijo cherry_pick_
    para activar la persistencia nativa de resultados -- hallazgo Fase 0.3).
    """

    def __init__(
        self,
        setup_name: str,
        active_roles: list[str],
        ticker: str,
        evaluation_start_date: pd.Timestamp,
    ):
        super().__init__()
        self.setup_name = setup_name
        self.active_roles = active_roles
        self.ticker = ticker
        self.evaluation_start_date = pd.Timestamp(evaluation_start_date)

        self._ohlcv_rows: list[dict] = []
        self._news_rows: list[dict] = []

    # -- Acumulación de buffers -------------------------------------------------

    def _append_price(self, date: pd.Timestamp, price_row: dict) -> None:
        # Campos ajustados (splits/dividendos), consistentes con los que
        # usa el framework para ejecutar las órdenes de verdad.
        self._ohlcv_rows.append(
            {
                "date": date,
                "open": price_row["adjusted_open"],
                "high": price_row["adjusted_high"],
                "low": price_row["adjusted_low"],
                "close": price_row["adjusted_close"],
                "volume": price_row["volume"],
            }
        )

    def _append_news(self, date: pd.Timestamp, news_texts: list[str]) -> None:
        for text in news_texts:
            self._news_rows.append({"date": date, "symbol": self.ticker, "news_text": text})

    def _ohlcv_history(self) -> pd.DataFrame:
        df = pd.DataFrame(self._ohlcv_rows)
        return df.set_index("date").sort_index()

    def _news_items(self) -> pd.DataFrame:
        # select_news() (agents/_common.py) espera columnas date/news_text
        # y filtra por ventana; symbol ya viene fijado a self.ticker.
        return pd.DataFrame(self._news_rows, columns=["date", "symbol", "news_text"])

    # -- Estado de posición (Fase 2.4: leído del framework, no inventado) -------

    def _position_state(self, framework) -> str:
        held = framework.portfolio.get(self.ticker, {}).get("quantity", 0)
        return "holding shares (fully invested)" if held > 0 else "no position (all cash)"

    # -- Traducción size -> quantity (Fase 2.4: en Python, no en el LLM) --------

    def _execute_decision(
        self, trader_output: dict, date: pd.Timestamp, framework, price: float
    ) -> None:
        action = trader_output["action"]
        size = trader_output["size"]

        if action == "buy":
            quantity = int(size * framework.cash / price)
            if quantity > 0:
                framework.buy(date, self.ticker, price, quantity)
        elif action == "sell":
            held = framework.portfolio.get(self.ticker, {}).get("quantity", 0)
            quantity = int(size * held)
            if quantity > 0:
                framework.sell(date, self.ticker, price, quantity)
        # action == "hold": no-op. _resolve_quantity de FINSABER-2 también
        # resuelve de forma segura cualquier orden redundante a quantity=0
        # (comprobado en Fase 2.4), pero no dependemos de ello aquí.

    # -- Callback principal -------------------------------------------------

    def on_data(self, date, today_data: dict, framework) -> None:
        # FINSABER-2 itera date_range con objetos datetime.date, no
        # pd.Timestamp (verificado en vivo, Fase 3.5) -- se normaliza aquí
        # una vez para que el resto de la clase (buffers, comparación con
        # evaluation_start_date, .date() en los prompts de los agentes)
        # trabaje siempre con pd.Timestamp.
        date = pd.Timestamp(date)

        price_row = today_data.get("price", {}).get(self.ticker)
        if price_row is None:
            logger.debug("%s/%s: sin dato de precio para %s, se omite el día", self.setup_name, self.ticker, date)
            return

        self._append_price(date, price_row)
        self._append_news(date, today_data.get("news", {}).get(self.ticker, []))

        if date < self.evaluation_start_date:
            # Preload: acumula buffers pero no opera. El runner (Fase 3.6)
            # es responsable de cargar suficientes sesiones previas para
            # que, en evaluation_start_date, ohlcv_history ya tenga >=
            # MIN_OHLCV_HISTORY_FOR_TECHNICAL filas si el rol técnico está activo.
            return

        ohlcv_history = self._ohlcv_history()

        if "technical" in self.active_roles and len(ohlcv_history) < MIN_OHLCV_HISTORY_FOR_TECHNICAL:
            # No debería ocurrir si el runner precargó suficiente historia;
            # si ocurre, es una configuración incorrecta del preload, no un
            # warmup esperado -- se loguea como warning, no debug.
            logger.warning(
                "%s/%s: %d/%d sesiones en evaluation_start_date (%s) -- "
                "preload insuficiente, revisar runner de Fase 3.6",
                self.setup_name, self.ticker, len(ohlcv_history),
                MIN_OHLCV_HISTORY_FOR_TECHNICAL, self.evaluation_start_date,
            )
            return

        news_items = self._news_items()
        position_state = self._position_state(framework)

        result = run_configuration(
            self.setup_name,
            self.active_roles,
            date,
            self.ticker,
            ohlcv_history,
            news_items,
            position_state,
        )

        if result.status == "failed":
            logger.warning(
                "%s/%s: día %s marcado como fallido (rol: %s)",
                self.setup_name, self.ticker, date, result.failed_role,
            )
            return

        self._execute_decision(result.trader_output, date, framework, price_row["adjusted_close"])
