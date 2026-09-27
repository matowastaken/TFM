"""
run_backtest.py

Runner de un backtest para una configuración de ablación (Fase 3.5/3.6).
Conecta FinsaberParquetDataset (FINSABER_DATA_PATH), fija log_base_dir
(FINSABER_RESULTS_PATH) e instancia FINSABER con la API real del paquete
(FINSABER(trade_config: dict).run_iterative_tickers(strategy_class,
strat_params, tickers)) -- confirmada en Fase 3.6 contra el código fuente
del paquete instalado, no solo la documentación.

Uso (smoke test de 3.6, 1 ticker, pocos días):
    python run_backtest.py --setup full_system --ticker NVDA \
        --eval-start 2024-08-01 --eval-end 2024-08-08 --preload-days 50

Notas de diseño:
  - `--preload-days` sesiones ANTES de --eval-start se cargan para que el
    rol técnico (>=50 sesiones, Fase 2.1) tenga histórico completo desde
    el primer día real de evaluación (Fase 3.5: evaluation_start_date).
  - La persistencia de resultados con el motor FINSABER (a diferencia de
    FINSABERBt) no depende del nombre del setup_name -- confirmado en
    Fase 3.6 contra el código fuente del paquete.
"""

from __future__ import annotations

import argparse
import logging
import os

import pandas as pd
from dotenv import load_dotenv

import backtest.finsaber as _finsaber_module
from backtest.data_util.finsaber_parquet_dataset import FinsaberParquetDataset
from backtest.finsaber import FINSABER

from strategy.ablation_strategy import AblationStrategy
from strategy.orchestrator import load_ablation_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _disable_optional_aggregation() -> None:
    """
    Workaround de dos bugs reales de finsaber==2.0.1 (verificados en
    Fase 3.6) en la llamada a aggregate_results_one_strategy() dentro de
    FINSABER.run_iterative_tickers():

    1) Se llama sin pasar output_dir, así que cae a una ruta relativa por
       defecto DENTRO del paquete instalado en vez de usar
       trade_config.log_base_dir, y esa ruta no existe -> FileNotFoundError.
    2) Aunque se le pase output_dir explícito (probado), la función usa
       internamente DataFrame._append(), un método privado de pandas
       eliminado en pandas >= 2.x (confirmado con pandas 3.0.2) ->
       AttributeError.

    aggregate_results_one_strategy() solo consolida resultados entre
    rolling windows -- no la usamos (nuestro diseño no tiene rolling
    windows, Fase 1.2: un único periodo de test) y la consolidación real
    entre las 5 configuraciones x 4 tickers se hace a mano en la Fase 4.3
    a partir de los CSV/JSON que write_result_artifacts() SÍ escribe
    correctamente (equity_curve.csv, metrics.json, trades.csv,
    rejected_orders.csv, llm_costs.csv -- verificado que se generan bien
    ANTES de este punto, con la estructura de carpetas documentada en
    Fase 0.3). Se desactiva sustituyéndola por un no-op.
    """
    _finsaber_module.aggregate_results_one_strategy = lambda *a, **kw: None


def build_trade_config(setup_name: str, ticker: str, preload_start: str, eval_end: str) -> dict:
    load_dotenv()
    data_path = os.environ["FINSABER_DATA_PATH"]
    results_path = os.environ["FINSABER_RESULTS_PATH"]

    data_loader = FinsaberParquetDataset(
        root=data_path,
        tickers=[ticker],
        modalities=("price", "news"),  # filing_k/filing_q no usados (Fase 2.3: fuera del prototipo inicial)
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


def run(setup_name: str, ticker: str, eval_start: str, eval_end: str, preload_days: int) -> dict:
    ablation_config = load_ablation_config()
    if setup_name not in ablation_config:
        raise ValueError(f"setup_name desconocido: {setup_name}. Configs válidas: {list(ablation_config)}")
    active_roles = ablation_config[setup_name]

    eval_start_ts = pd.Timestamp(eval_start)
    # Aproximación de calendario: preload_days sesiones ~ preload_days *
    # 1.5 días naturales (fines de semana/festivos), con margen.
    preload_start = (eval_start_ts - pd.Timedelta(days=int(preload_days * 1.6))).strftime("%Y-%m-%d")

    trade_config = build_trade_config(setup_name, ticker, preload_start, eval_end)
    _disable_optional_aggregation()
    app = FINSABER(trade_config)

    strat_params = {
        "setup_name": setup_name,
        "active_roles": active_roles,
        "ticker": "$symbol",
        "evaluation_start_date": eval_start_ts,
    }

    logger.info(
        "Backtest %s / %s: preload desde %s, evaluación real %s -> %s",
        setup_name, ticker, preload_start, eval_start, eval_end,
    )
    metrics = app.run_iterative_tickers(AblationStrategy, strat_params=strat_params, tickers=[ticker])
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test / backtest de una configuración de ablación (Fase 3.6)")
    parser.add_argument("--setup", required=True, help="setup_name de config/ablation.yaml")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--eval-start", required=True, help="Primer día real de evaluación (YYYY-MM-DD)")
    parser.add_argument("--eval-end", required=True, help="Último día del backtest (YYYY-MM-DD)")
    parser.add_argument("--preload-days", type=int, default=50, help="Sesiones de histórico antes de --eval-start")
    args = parser.parse_args()

    metrics = run(args.setup, args.ticker, args.eval_start, args.eval_end, args.preload_days)

    # run_iterative_tickers() devuelve {periodo: {ticker: metrics}} cuando
    # save_results=True (nuestro caso), no {ticker: metrics} directamente
    # -- verificado en vivo, Fase 3.6.
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