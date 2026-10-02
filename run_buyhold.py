"""run_buyhold.py — Fase 4.3: benchmark externo buy-and-hold por ticker.

Se ejecuta en el MISMO motor (FINSABER + FinsaberParquetDataset) y con las mismas convenciones que
las 20 configuraciones de ablación, para que sea comparable:
  - ventana de evaluación 2024-08-01 -> 2025-01-31 (sin preload: no hace falta histórico)
  - la señal de compra se emite el primer día (2024-08-01) y se ejecuta a la apertura del siguiente
    (execution_timing="next_open"), igual que la primera orden del resto de configuraciones
  - capital inicial 100.000, comisiones y liquidación final del motor
  - no hay llamadas a LLM (coste LLM = 0)

Resultados en <FINSABER_RESULTS_PATH>/buy_and_hold/BuyAndHoldIso/<periodo>/<ticker>/ (mismo formato).
Requiere ejecutarse desde la raíz del repo (importa run_backtest).

Uso:  python run_buyhold.py [--tickers JPM JNJ XOM NVDA] [--eval-start 2024-08-01] [--eval-end 2025-01-31]
"""
import argparse
import logging
import os
from pathlib import Path

from backtest.finsaber import FINSABER
from backtest.strategy.timing_llm import BaseStrategyIso

from run_backtest import _disable_optional_aggregation, _patch_equity_plot, build_trade_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SETUP_NAME = "buy_and_hold"


class BuyAndHoldIso(BaseStrategyIso):
    """Compra con todo el efectivo el primer día y mantiene hasta el final (el motor liquida al cierre)."""

    def __init__(self, ticker: str):
        super().__init__()
        self.ticker = ticker
        self._bought = False

    def on_data(self, date, today_data, framework):
        if self._bought:
            return
        bar = today_data.get("price", {}).get(self.ticker)
        if bar is None:
            return  # sin dato de precio ese día: reintenta al siguiente
        framework.buy(date, self.ticker, bar["adjusted_close"], -1)  # -1 = todo el cash
        self._bought = True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=["JPM", "JNJ", "XOM", "NVDA"])
    ap.add_argument("--eval-start", default="2024-08-01")
    ap.add_argument("--eval-end", default="2025-01-31")
    args = ap.parse_args()

    _disable_optional_aggregation()
    for ticker in args.tickers:
        logger.info("Buy-and-hold %s: %s -> %s", ticker, args.eval_start, args.eval_end)
        cfg = build_trade_config(SETUP_NAME, ticker, args.eval_start, args.eval_end)
        _patch_equity_plot(Path(cfg["log_base_dir"]) / "_figures", show=False)
        app = FINSABER(cfg)
        res = app.run_iterative_tickers(BuyAndHoldIso, strat_params={"ticker": "$symbol"}, tickers=[ticker])
        m = next(iter(res.values()))[ticker] if res else {}
        logger.info("  %s total_return=%s max_drawdown=%s", ticker, m.get("total_return"), m.get("max_drawdown"))


if __name__ == "__main__":
    main()
