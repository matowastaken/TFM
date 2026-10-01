"""
run_backtest.py

Runner de un backtest para una configuración de ablación (Fase 3.5/3.6 y
4.1). Conecta FinsaberParquetDataset (FINSABER_DATA_PATH), fija
log_base_dir (FINSABER_RESULTS_PATH) e instancia FINSABER con la API real
del paquete (FINSABER(trade_config: dict).run_iterative_tickers(
strategy_class, strat_params, tickers)), confirmada contra el código
fuente de finsaber==2.0.1.

Uso (un backtest, con recuperación ante fallos):
    python run_backtest.py --setup full_system --ticker NVDA \
        --eval-start 2024-08-01 --eval-end 2024-08-08 --preload-days 50

Notas de diseño:
  - `--preload-days` sesiones ANTES de --eval-start se cargan para que el
    rol técnico (>=50 sesiones, Fase 2.1) tenga histórico completo desde
    el primer día real de evaluación (Fase 3.5: evaluation_start_date).
  - La persistencia de resultados con el motor FINSABER no depende del
    nombre del setup_name (confirmado en Fase 3.6).

Fase 4.1 -- recuperación ante fallos:
  run_with_recovery() ejecuta el backtest; si un día falla (DayFailure, ver
  strategy/ablation_strategy.py), borra los artefactos parciales de ese
  (config, ticker) y lo relanza con attempt+1. La estrategia reproduce
  desde el log los días ya completados (sin llamar al LLM) y continúa en el
  día fallido. A los MAX_DAY_FAILURES fallos consecutivos en el mismo día
  se lanza ManualReviewRequired y se detiene.

  Figuras: FINSABER llama a plt.show() (bloqueante) al terminar cada
  backtest. _patch_equity_plot() lo sustituye: guarda el PNG en
  <FINSABER_RESULTS_PATH>/_figures/ sin detener la ejecución.

  Se decide si hubo fallo consultando failures.jsonl (no solo por el tipo
  de excepción), de modo que funciona aunque FINSABER envolviera la
  excepción original.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

import backtest.finsaber as _finsaber_module
from backtest.data_util.finsaber_parquet_dataset import FinsaberParquetDataset
from backtest.finsaber import FINSABER

from llm.call_llm import configure_call_log
from strategy.ablation_strategy import (
    AblationStrategy,
    days_log_path,
    failures_log_path,
    jsonl_read,
)
from strategy.orchestrator import load_ablation_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_DAY_FAILURES = 3


class ManualReviewRequired(RuntimeError):
    """El mismo día falló MAX_DAY_FAILURES veces seguidas: revisión manual."""


def _disable_optional_aggregation() -> None:
    """
    Workaround de dos bugs reales de finsaber==2.0.1 (verificados en
    Fase 3.6) en aggregate_results_one_strategy(), llamada al final de
    FINSABER.run_iterative_tickers():

    1) Se llama sin output_dir y cae a una ruta relativa dentro del paquete
       instalado -> FileNotFoundError.
    2) Usa DataFrame._append(), eliminado en pandas >= 2.x -> AttributeError.

    Solo consolida resultados entre rolling windows, que este proyecto no
    usa (Fase 1.2); la consolidación real se hace a mano en la Fase 4.3 a
    partir de los artefactos que write_result_artifacts() sí escribe. Se
    sustituye por un no-op.
    """
    _finsaber_module.aggregate_results_one_strategy = lambda *a, **kw: None


def _patch_equity_plot(figures_dir: Path, show: bool = False) -> None:
    """
    Sustituye FINSABER._plot_equity_curve (finsaber.py, ~l.125), que termina
    en plt.show() BLOQUEANTE: la ejecución se detenía hasta cerrar la ventana.

    Versión parcheada:
      - guarda SIEMPRE el PNG en <figures_dir>/<setup>__<ticker>.png
        (se sobrescribe en reintentos; no lo borra _clear_partial_results);
      - show=False (por defecto): dibuja con la API orientada a objetos
        (Figure + canvas Agg), sin abrir ventana y sin estado global de pyplot;
      - show=True: ventana NO bloqueante (plt.show(block=False) + pause corto),
        sin detener la ejecución; las ventanas quedan abiertas;
      - un fallo al dibujar nunca tumba el backtest (se registra un warning).
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    def _plot(self, equity_with_time, ticker):
        try:
            setup = getattr(self.trade_config, "setup_name", "setup")
            figures_dir.mkdir(parents=True, exist_ok=True)
            out = figures_dir / f"{setup}__{ticker}.png"

            if show:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(10, 6))
            else:
                fig = Figure(figsize=(10, 6))
                FigureCanvasAgg(fig)
                ax = fig.subplots()

            ax.plot(equity_with_time["datetime"], equity_with_time["equity"], label="Equity Curve")
            ax.set_title(f"Equity Curve for {ticker} - {setup}")
            ax.set_xlabel("Date")
            ax.set_ylabel("Equity")
            ax.grid(alpha=0.3)
            ax.legend()
            fig.autofmt_xdate()
            fig.savefig(out, dpi=150, bbox_inches="tight")
            logger.info("Figura guardada: %s", out)

            if show:
                plt.show(block=False)
                plt.pause(0.1)
        except Exception as e:  # noqa: BLE001
            logger.warning("No se pudo guardar/mostrar la equity curve de %s: %s", ticker, e)

    FINSABER._plot_equity_curve = _plot


def default_run_log_dir() -> Path:
    load_dotenv()
    return Path(os.environ["FINSABER_RESULTS_PATH"]) / "_run_logs"


def build_trade_config(setup_name: str, ticker: str, preload_start: str, eval_end: str) -> dict:
    load_dotenv()
    data_path = os.environ["FINSABER_DATA_PATH"]
    results_path = os.environ["FINSABER_RESULTS_PATH"]

    data_loader = FinsaberParquetDataset(
        root=data_path,
        tickers=[ticker],
        modalities=("price", "news"),  # filing_k/filing_q no usados (Fase 2.3)
    )

    return {
        "tickers": [ticker],
        "date_from": preload_start,  # incluye el preload de --preload-days
        "date_to": eval_end,
        "cash": 100000.0,
        "execution_timing": "next_open",
        "setup_name": setup_name,
        "log_base_dir": results_path,
        "data_loader": data_loader,
        "save_results": True,
        "silence": False,
    }


def _run_once(
    setup_name: str,
    ticker: str,
    eval_start: str,
    eval_end: str,
    preload_days: int,
    run_log_dir: Path,
    attempt: int,
    show_figures: bool = False,
) -> dict:
    ablation_config = load_ablation_config()
    if setup_name not in ablation_config:
        raise ValueError(f"setup_name desconocido: {setup_name}. Configs válidas: {list(ablation_config)}")
    active_roles = ablation_config[setup_name]

    eval_start_ts = pd.Timestamp(eval_start)
    # preload_days sesiones ~ preload_days * 1.6 días naturales, con margen.
    preload_start = (eval_start_ts - pd.Timedelta(days=int(preload_days * 1.6))).strftime("%Y-%m-%d")

    trade_config = build_trade_config(setup_name, ticker, preload_start, eval_end)
    _disable_optional_aggregation()
    _patch_equity_plot(Path(trade_config["log_base_dir"]) / "_figures", show=show_figures)
    app = FINSABER(trade_config)

    strat_params = {
        "setup_name": setup_name,
        "active_roles": active_roles,
        "ticker": "$symbol",
        "evaluation_start_date": eval_start_ts,
        "run_log_dir": str(run_log_dir),
        "attempt": attempt,
    }

    logger.info(
        "Backtest %s / %s (intento %d): preload desde %s, evaluación real %s -> %s",
        setup_name, ticker, attempt, preload_start, eval_start, eval_end,
    )
    return app.run_iterative_tickers(AblationStrategy, strat_params=strat_params, tickers=[ticker])


def _clear_partial_results(results_path: str, setup_name: str, ticker: str) -> None:
    """Borra <log_base_dir>/<setup>/<strategy>/<periodo>/<ticker>/ de intentos previos."""
    for p in Path(results_path, setup_name).glob(f"*/*/{ticker}"):
        shutil.rmtree(p, ignore_errors=True)


def _pair_failures(run_log_dir: Path, setup_name: str, ticker: str) -> list[dict]:
    return [
        r for r in jsonl_read(failures_log_path(run_log_dir))
        if r["setup"] == setup_name and r["ticker"] == ticker
    ]


def _last_attempt(run_log_dir: Path, setup_name: str, ticker: str) -> int:
    attempts = [r.get("attempt", 0) for r in _pair_failures(run_log_dir, setup_name, ticker)]
    attempts += [r.get("attempt", 0) for r in jsonl_read(days_log_path(run_log_dir, setup_name, ticker))]
    return max(attempts, default=0)


def run_with_recovery(
    setup_name: str,
    ticker: str,
    eval_start: str,
    eval_end: str,
    preload_days: int = 50,
    run_log_dir: Path | None = None,
    resume: bool = False,
    max_day_failures: int = MAX_DAY_FAILURES,
    show_figures: bool = False,
) -> dict:
    """
    Ejecuta un (config, ticker) hasta completarlo sin días fallidos.

    resume=False (por defecto): empieza limpio -- borra el log de días de
        este (config, ticker), para que un backtest nuevo NO reproduzca
        outputs de una ejecución anterior.
    resume=True: conserva el log de días (p. ej. tras un
        ManualReviewRequired ya revisado) y retoma desde el último día
        completado.
    """
    run_log_dir = Path(run_log_dir) if run_log_dir else default_run_log_dir()
    run_log_dir.mkdir(parents=True, exist_ok=True)
    results_path = os.environ["FINSABER_RESULTS_PATH"]
    configure_call_log(run_log_dir / "calls.jsonl")

    if not resume:
        days_log_path(run_log_dir, setup_name, ticker).unlink(missing_ok=True)

    attempt = _last_attempt(run_log_dir, setup_name, ticker)
    last_fail_date: str | None = None
    consecutive = 0

    while True:
        attempt += 1
        _clear_partial_results(results_path, setup_name, ticker)

        metrics, error = None, None
        try:
            metrics = _run_once(
                setup_name, ticker, eval_start, eval_end, preload_days, run_log_dir, attempt,
                show_figures=show_figures,
            )
        except Exception as e:  # noqa: BLE001 -- se decide abajo según failures.jsonl
            error = e

        fails = [
            r for r in _pair_failures(run_log_dir, setup_name, ticker)
            if r.get("attempt") == attempt
        ]
        if not fails:
            if error is not None:
                raise error  # error ajeno a DayFailure (p. ej. divergencia de replay)
            return metrics

        fail_date = fails[0]["date"]
        consecutive = consecutive + 1 if fail_date == last_fail_date else 1
        last_fail_date = fail_date
        logger.warning(
            "%s/%s: fallo el %s (rol %s), fallos consecutivos ese día: %d/%d",
            setup_name, ticker, fail_date, fails[0]["failed_role"], consecutive, max_day_failures,
        )
        if consecutive >= max_day_failures:
            _clear_partial_results(results_path, setup_name, ticker)
            raise ManualReviewRequired(
                f"{setup_name}/{ticker}: {consecutive} fallos consecutivos el {fail_date} "
                f"(último rol: {fails[0]['failed_role']}). Revisar {failures_log_path(run_log_dir)} "
                f"y relanzar con --resume."
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest de una configuración de ablación, con recuperación ante fallos")
    parser.add_argument("--setup", required=True, help="setup_name de config/ablation.yaml")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--eval-start", required=True, help="Primer día real de evaluación (YYYY-MM-DD)")
    parser.add_argument("--eval-end", required=True, help="Último día del backtest (YYYY-MM-DD)")
    parser.add_argument("--preload-days", type=int, default=50, help="Sesiones de histórico antes de --eval-start")
    parser.add_argument("--resume", action="store_true", help="Conservar el log de días y retomar (tras revisión manual)")
    parser.add_argument("--run-log-dir", default=None, help="Carpeta de logs (por defecto FINSABER_RESULTS_PATH/_run_logs)")
    parser.add_argument("--show-figures", action="store_true",
                        help="Además de guardar el PNG, abrir la ventana sin bloquear la ejecución")
    args = parser.parse_args()

    metrics = run_with_recovery(
        args.setup, args.ticker, args.eval_start, args.eval_end, args.preload_days,
        run_log_dir=args.run_log_dir, resume=args.resume, show_figures=args.show_figures,
    )

    # run_iterative_tickers() devuelve {periodo: {ticker: metrics}} con save_results=True.
    per_ticker_metrics = next(iter(metrics.values())) if metrics else {}

    for ticker, ticker_metrics in per_ticker_metrics.items():
        logger.info("=== %s ===", ticker)
        for key in ("total_return", "sharpe_ratio", "max_drawdown", "total_llm_cost"):
            if key in ticker_metrics:
                logger.info("  %s: %s", key, ticker_metrics[key])
        logger.info("  trades: %d", len(ticker_metrics.get("trades", [])))
        logger.info("  rejected_orders: %d", len(ticker_metrics.get("rejected_orders", [])))


if __name__ == "__main__":
    main()