"""consolidate_results.py — Fase 4.3: consolida los 20 backtests (+ buy-and-hold opcional).

Entradas:
  <results>/<setup>/AblationStrategy/<periodo>/<ticker>/{equity_curve.csv,metrics.json,trades.csv,...}
  <results>/_run_logs/calls.jsonl                    (coste LLM real, fuente de verdad)
  <results>/buy_and_hold/*/<periodo>/<ticker>/...    (opcional, de run_buyhold.py)

Decisiones:
  * Las métricas se RECALCULAN desde equity_curve.csv restringido a la ventana de evaluación
    (>= --eval-start). FINSABER las calcula sobre toda la ventana cargada, que incluye ~55
    sesiones de preload con equity plana: eso diluye annual_return, volatilidad, Sharpe y Sortino
    (total_return y max_drawdown no cambian). Las fórmulas replican las de FINSABER
    (backtest_framework_iso.evaluate), solo cambia la ventana.
  * Calmar = annual_return / max_drawdown (fracción). NaN si el drawdown es 0.
  * Coste LLM: suma de calls.jsonl por par (no total_llm_cost de metrics.json).
  * Nº de operaciones: filas de trades.csv.
  * Deltas vs full_system: puntos porcentuales para retornos y drawdown, diferencia absoluta
    para ratios, nº de operaciones y coste.

Uso:
  python consolidate_results.py --results C:\\finsaber-data\\results --out consolidated
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SETUPS = ["baseline_price_only", "full_system", "no_technical", "no_sentiment", "no_fundamental"]
TICKERS = ["JPM", "JNJ", "XOM", "NVDA"]
BH = "buy_and_hold"
RF = 0.03  # risk_free_rate de run_config.json de FINSABER


def eval_metrics(equity: pd.Series, rf: float = RF) -> dict:
    """Mismas fórmulas que FINSABER, sobre la serie ya recortada a la ventana de evaluación."""
    eq = equity.astype(float).reset_index(drop=True)
    r = eq.pct_change().dropna()
    total = eq.iloc[-1] / eq.iloc[0] - 1
    ann = (1 + total) ** (252 / len(eq)) - 1
    vol = r.std() * np.sqrt(252)
    down = r[r < 0]
    dd_dev = down.std() * np.sqrt(252) if len(down) > 1 else 0
    sharpe = (r.mean() * 252 - rf) / vol if vol > 0 else 0.0
    sortino = (r.mean() * 252 - rf) / dd_dev if dd_dev > 0 else 0.0
    peak = eq.cummax()
    mdd = abs(((eq - peak) / peak).min())  # fracción positiva
    return {
        "total_return": total, "annual_return": ann, "annual_volatility": vol,
        "sharpe": sharpe, "sortino": sortino,
        "max_drawdown": mdd, "calmar": ann / mdd if mdd > 0 else np.nan,
        "final_value": eq.iloc[-1],
    }


def find_pair_dir(results: Path, setup: str, ticker: str):
    m = sorted(results.glob(f"{setup}/*/*/{ticker}"))
    return m[-1] if m else None


def load_equity(pair_dir: Path, eval_start: str) -> pd.Series:
    e = pd.read_csv(pair_dir / "equity_curve.csv", parse_dates=["datetime"])
    e = e[e["datetime"] >= pd.Timestamp(eval_start)]
    return e.set_index("datetime")["equity"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", default="consolidated")
    ap.add_argument("--eval-start", default="2024-08-01")
    ap.add_argument("--calls", default=None, help="por defecto <results>/_run_logs/calls.jsonl")
    args = ap.parse_args()

    results = Path(args.results)
    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    calls_path = Path(args.calls) if args.calls else results / "_run_logs" / "calls.jsonl"
    calls = pd.read_json(calls_path, lines=True)
    cost = calls.groupby(["setup", "ticker"])["cost"].sum()
    ncalls = calls.groupby(["setup", "ticker"]).size()

    rows, curves = [], {}
    for ticker in TICKERS:
        for setup in SETUPS + [BH]:
            d = find_pair_dir(results, setup, ticker)
            if d is None:
                if setup != BH:
                    print(f"AVISO: falta {setup}/{ticker}")
                continue
            eq = load_equity(d, args.eval_start)
            m = eval_metrics(eq)
            fw = json.loads((d / "metrics.json").read_text())
            trades = d / "trades.csv"
            m.update({
                "setup": setup, "ticker": ticker, "days": len(eq),
                "n_trades": max(sum(1 for _ in open(trades, encoding="utf-8")) - 1, 0) if trades.exists() else 0,
                "commission": fw.get("total_commission", 0.0),
                "llm_cost_usd": float(cost.get((setup, ticker), 0.0)),
                "llm_calls": int(ncalls.get((setup, ticker), 0)),
                "finsaber_total_return": fw["total_return"],
                "finsaber_sharpe": fw["sharpe_ratio"],
            })
            rows.append(m)
            curves[(setup, ticker)] = eq

    df = pd.DataFrame(rows)
    cols = ["setup", "ticker", "days", "total_return", "annual_return", "annual_volatility", "sharpe",
            "sortino", "max_drawdown", "calmar", "n_trades", "commission", "llm_calls", "llm_cost_usd",
            "final_value", "finsaber_total_return", "finsaber_sharpe"]
    df = df[cols]
    df.to_csv(out / "consolidated_results.csv", index=False)

    # Control: total_return recalculado debe coincidir con el de FINSABER
    diff = (df["total_return"] - df["finsaber_total_return"]).abs().max()
    print(f"Control total_return recalculado vs FINSABER: max|diff| = {diff:.2e}")

    abl = df[df["setup"].isin(SETUPS)]
    metrics = ["total_return", "annual_return", "sharpe", "calmar", "max_drawdown", "n_trades", "llm_cost_usd"]

    # Promedio entre tickers por configuración
    summary = abl.groupby("setup")[metrics].mean().reindex(SETUPS)
    summary.to_csv(out / "summary_by_config.csv")

    # Deltas vs full_system (global y por ticker)
    pp = ["total_return", "annual_return", "max_drawdown"]
    def deltas(frame):
        base = frame.loc["full_system"]
        d = frame - base
        d[pp] = d[pp] * 100  # puntos porcentuales
        return d
    deltas(summary).to_csv(out / "deltas_vs_full_global.csv")
    per_t = []
    for t in TICKERS:
        s = abl[abl.ticker == t].set_index("setup")[metrics].reindex(SETUPS)
        dd = deltas(s); dd.insert(0, "ticker", t); per_t.append(dd.reset_index())
    pd.concat(per_t).to_csv(out / "deltas_vs_full_by_ticker.csv", index=False)

    # Tabla ancha retorno total por ticker x config
    wide = df.pivot(index="ticker", columns="setup", values="total_return")
    wide.to_csv(out / "total_return_ticker_x_config.csv")

    # Figuras: curvas de equity por ticker (5 configs + B&H si existe)
    colors = {"baseline_price_only": "#888888", "full_system": "#1f77b4", "no_technical": "#d62728",
              "no_sentiment": "#2ca02c", "no_fundamental": "#ff7f0e", BH: "#000000"}
    for t in TICKERS:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        for s in SETUPS + [BH]:
            if (s, t) in curves:
                c = curves[(s, t)]
                ax.plot(c.index, c.values / c.iloc[0] * 100, label=s, color=colors[s],
                        ls="--" if s == BH else "-", lw=1.6 if s == "full_system" else 1.2)
        ax.set_title(f"Equity normalizada (base 100) - {t}")
        ax.set_ylabel("Equity (base 100)"); ax.grid(alpha=.3); ax.legend(fontsize=8)
        fig.autofmt_xdate(); fig.savefig(out / "figures" / f"equity_{t}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    pd.set_option("display.width", 220)
    print("\nPromedio entre tickers por configuración:\n", summary.round(4).to_string())
    print("\nRetorno total por ticker x config:\n", wide.round(4).to_string())
    print(f"\nSalida en: {out.resolve()}")


if __name__ == "__main__":
    main()
