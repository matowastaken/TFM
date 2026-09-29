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
  - Warmup (rediseñado en Fase 3.5 a partir de feedback): el runner
    (Fase 3.6) carga el dataset ~50 sesiones ANTES de la fecha real de
    evaluación. Esta clase recibe explícitamente evaluation_start_date:
    acumula OHLCV/noticias desde el primer día cargado, pero NO llama al
    orquestador ni opera hasta date >= evaluation_start_date (Fase 1.2).
  - Precios: los indicadores técnicos y el precio de referencia pasado a
    framework.buy()/.sell() usan los campos ajustados (adjusted_*), igual
    que la ejecución real de órdenes (fill a next-open contra
    adjusted_open, confirmado en backtest_framework_iso.py).
  - Bug de coste (Fase 0.4) corregido de verdad en Fase 3.5: ver
    llm/call_llm.py.

Fase 4.1 -- recuperación ante fallos (pausa/reanudación con replay):
  - FINSABER NO captura excepciones de on_data() (verificado en el código
    fuente de finsaber==2.0.1: la llamada strategy.on_data(...) en
    backtest_framework_iso.py no está rodeada por ningún try/except, y los
    except de finsaber.py solo cubren InsufficientTrainingDataException y
    (AttributeError, IndexError) en otros puntos). Una excepción lanzada
    aquí sale de framework.run() y de run_iterative_tickers().
  - Si un día falla (el orquestador devuelve status "failed"), se registra
    en <run_log_dir>/failures.jsonl (con los informes parciales, que NO se
    reutilizan) y se lanza DayFailure. Tras el primer fallo la instancia
    queda "abortada": cualquier on_data() posterior relanza el error sin
    llamar al LLM (defensa extra por si el comportamiento de FINSABER
    cambiase en otra versión).
  - Cada día completado con éxito se añade a
    <run_log_dir>/<setup>__<ticker>.days.jsonl (informes de analista +
    decisión del trader + estado de posición). Este fichero es el
    registro de outputs por ticker, rol y fecha, y la fuente del replay.
  - run_backtest.run_with_recovery() relanza el (config, ticker) desde el
    principio con attempt+1. Los días ya presentes en el log se
    REPRODUCEN sin llamar al LLM: FINSABER reconstruye cartera y órdenes
    con la decisión guardada. Desde el día fallido se llama al LLM de
    nuevo. Se comprueba que la posición en el replay coincide con la
    guardada; si no, se aborta (divergencia).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pandas as pd
from finsaber.strategy.timing_llm import BaseStrategyIso

from llm.call_llm import set_call_context
from strategy.orchestrator import run_configuration

logger = logging.getLogger(__name__)

MIN_OHLCV_HISTORY_FOR_TECHNICAL = 50  # SMA50/ATR14 (Fase 2.1)


class DayFailure(Exception):
    """Un día no pudo completarse (agente sin respuesta válida)."""

    def __init__(self, date, failed_role, setup_name, ticker):
        self.date = pd.Timestamp(date)
        self.failed_role = failed_role
        self.setup_name = setup_name
        self.ticker = ticker
        super().__init__(
            f"{setup_name}/{ticker}: fallo el {self.date.date()} (rol: {failed_role})"
        )


def days_log_path(run_log_dir: str | Path, setup_name: str, ticker: str) -> Path:
    return Path(run_log_dir) / f"{setup_name}__{ticker}.days.jsonl"


def failures_log_path(run_log_dir: str | Path) -> Path:
    return Path(run_log_dir) / "failures.jsonl"


def jsonl_append(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def jsonl_read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


class AblationStrategy(BaseStrategyIso):
    """
    Una instancia por (setup_name, ticker). setup_name identifica la
    configuración de ablación (Fase 1.3/3.4): full_system, no_technical,
    no_sentiment, no_fundamental, baseline_price_only.
    """

    def __init__(
        self,
        setup_name: str,
        active_roles: list[str],
        ticker: str,
        evaluation_start_date: pd.Timestamp,
        run_log_dir: str | None = None,
        attempt: int = 1,
    ):
        super().__init__()
        self.setup_name = setup_name
        self.active_roles = active_roles
        self.ticker = ticker
        self.evaluation_start_date = pd.Timestamp(evaluation_start_date)
        self.run_log_dir = Path(run_log_dir) if run_log_dir else None
        self.attempt = attempt

        self._ohlcv_rows: list[dict] = []
        self._news_rows: list[dict] = []

        self._failure: DayFailure | None = None
        self._replay: dict[str, dict] = {}
        if self.run_log_dir is not None:
            records = jsonl_read(days_log_path(self.run_log_dir, setup_name, ticker))
            self._replay = {r["date"]: r for r in records if r.get("status") == "ok"}
            if self._replay:
                logger.info(
                    "%s/%s (intento %d): %d días disponibles para replay",
                    setup_name, ticker, attempt, len(self._replay),
                )

    # -- Acumulación de buffers -------------------------------------------------

    def _append_price(self, date: pd.Timestamp, price_row: dict) -> None:
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

        logger.info(
            "%s/%s %s: trader decide %s (size=%.2f, confidence=%.2f) -- %s",
            self.setup_name, self.ticker, date.date(), action, size,
            trader_output.get("confidence", float("nan")), trader_output.get("rationale", ""),
        )

        if action == "buy":
            quantity = int(size * framework.cash / price)
            if quantity > 0:
                framework.buy(date, self.ticker, price, quantity)
        elif action == "sell":
            held = framework.portfolio.get(self.ticker, {}).get("quantity", 0)
            quantity = int(size * held)
            if quantity > 0:
                framework.sell(date, self.ticker, price, quantity)
        # action == "hold": no-op.

    # -- Fallos (Fase 4.1) ------------------------------------------------------

    def _forced_failure(self, date: pd.Timestamp) -> bool:
        """
        Hook SOLO para probar el bucle de recuperación: con
        TFM_FORCE_FAIL_DATE=YYYY-MM-DD, ese día falla en los primeros
        TFM_FORCE_FAIL_ATTEMPTS intentos (por defecto 1).
        """
        forced = os.environ.get("TFM_FORCE_FAIL_DATE")
        if not forced or date.strftime("%Y-%m-%d") != forced:
            return False
        return self.attempt <= int(os.environ.get("TFM_FORCE_FAIL_ATTEMPTS", "1"))

    def _fail(self, date: pd.Timestamp, failed_role: str, partial_reports) -> None:
        self._failure = DayFailure(date, failed_role, self.setup_name, self.ticker)
        if self.run_log_dir is not None:
            jsonl_append(
                failures_log_path(self.run_log_dir),
                {
                    "setup": self.setup_name,
                    "ticker": self.ticker,
                    "date": date.strftime("%Y-%m-%d"),
                    "attempt": self.attempt,
                    "failed_role": failed_role,
                    "partial_reports": partial_reports,
                },
            )
        logger.warning("%s", self._failure)
        raise self._failure

    # -- Callback principal -------------------------------------------------

    def on_data(self, date, today_data: dict, framework) -> None:
        if self._failure is not None:
            raise self._failure

        # FINSABER-2 itera con datetime.date; se normaliza a pd.Timestamp.
        date = pd.Timestamp(date)

        price_row = today_data.get("price", {}).get(self.ticker)
        if price_row is None:
            logger.debug("%s/%s: sin dato de precio para %s, se omite el día", self.setup_name, self.ticker, date)
            return

        self._append_price(date, price_row)
        self._append_news(date, today_data.get("news", {}).get(self.ticker, []))

        if date < self.evaluation_start_date:
            # Preload: acumula buffers pero no opera.
            return

        ohlcv_history = self._ohlcv_history()

        if "technical" in self.active_roles and len(ohlcv_history) < MIN_OHLCV_HISTORY_FOR_TECHNICAL:
            logger.warning(
                "%s/%s: %d/%d sesiones en evaluation_start_date (%s) -- "
                "preload insuficiente, revisar runner de Fase 3.6",
                self.setup_name, self.ticker, len(ohlcv_history),
                MIN_OHLCV_HISTORY_FOR_TECHNICAL, self.evaluation_start_date,
            )
            return

        position_state = self._position_state(framework)
        date_key = date.strftime("%Y-%m-%d")
        ref_price = price_row["adjusted_close"]

        # -- Replay: día ya completado en un intento anterior --------------
        replayed = self._replay.get(date_key)
        if replayed is not None:
            if replayed.get("position_state") != position_state:
                raise RuntimeError(
                    f"{self.setup_name}/{self.ticker} {date_key}: divergencia en replay "
                    f"(posición guardada={replayed.get('position_state')!r}, "
                    f"actual={position_state!r})"
                )
            self._execute_decision(replayed["trader_output"], date, framework, ref_price)
            return

        # -- Día nuevo: llamada real a los agentes --------------------------
        if self._forced_failure(date):
            self._fail(date, "forced_test", None)

        # Contexto para el log de llamadas (tokens/coste por llamada).
        set_call_context(
            setup=self.setup_name, ticker=self.ticker, date=date_key, attempt=self.attempt
        )

        result = run_configuration(
            self.setup_name,
            self.active_roles,
            date,
            self.ticker,
            ohlcv_history,
            self._news_items(),
            position_state,
        )

        if result.status == "failed":
            self._fail(date, str(result.failed_role), result.analyst_reports)

        if self.run_log_dir is not None:
            jsonl_append(
                days_log_path(self.run_log_dir, self.setup_name, self.ticker),
                {
                    "date": date_key,
                    "status": "ok",
                    "attempt": self.attempt,
                    "position_state": position_state,
                    "analyst_reports": result.analyst_reports,
                    "trader_output": result.trader_output,
                },
            )

        self._execute_decision(result.trader_output, date, framework, ref_price)