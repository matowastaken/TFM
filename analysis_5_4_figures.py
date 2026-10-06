"""Fase 5.4 - Figuras y tabla resumen finales para la memoria.
Uso (PowerShell):
  python analysis_5_4_figures.py --consolidated analysis\\replicas\\rep1\\consolidated_results.csv analysis\\replicas\\rep2\\consolidated_results.csv analysis\\replicas\\rep3\\consolidated_results.csv `
     --trades-nvda analysis\\trades_nvda --trades-xom analysis\\trades_xom --out analysis\\5_4
Genera: summary_3runs.csv/.md, risk_return.png, positions_NVDA.png, positions_XOM.png.
Las curvas de equity ya las genera consolidate_results.py en <out>\\figures\\equity_<ticker>.png de cada réplica (no se rehacen aquí).
Todo se calcula desde los ficheros indicados.
"""
import argparse, glob, os, re
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--consolidated", nargs=3, required=True)
ap.add_argument("--trades-nvda", required=True)
ap.add_argument("--trades-xom", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
RUNS = ["rep1", "rep2", "rep3"]; SETUPS = ["full_system", "no_technical", "no_sentiment", "no_fundamental"]
COL = {"full_system": "#1f77b4", "no_technical": "#d62728", "no_sentiment": "#2ca02c", "no_fundamental": "#ff7f0e"}
LAB = {"full_system": "full_system", "no_technical": "sin técnico", "no_sentiment": "sin sentimiento", "no_fundamental": "sin fundamental"}

allc = pd.concat([pd.read_csv(p).assign(run=r) for r, p in zip(RUNS, a.consolidated)], ignore_index=True)
bh = allc[(allc.setup == "buy_and_hold") & (allc.run == "rep1")].set_index("ticker")
D = allc[allc.setup.isin(SETUPS)].copy()
for c in ("total_return", "max_drawdown", "annual_return"): D[c] *= 100
bh_ret, bh_dd, bh_sh = bh.total_return.mean() * 100, bh.max_drawdown.mean() * 100, bh.sharpe.mean()

# ---- tabla resumen: media de 4 tickers por ejecución; media y SD entre las 3 ejecuciones ----
per_run = D.groupby(["run", "setup"]).agg(total_return=("total_return", "mean"), sharpe=("sharpe", "mean"),
          max_drawdown=("max_drawdown", "mean"), n_trades=("n_trades", "mean"), llm_cost=("llm_cost_usd", "sum")).reset_index()
rows = []
for s in SETUPS:
    x = per_run[per_run.setup == s]
    row = {"configuración": s}
    for m in ["total_return", "sharpe", "max_drawdown", "n_trades", "llm_cost"]:
        row[f"{m}_media"] = x[m].mean(); row[f"{m}_sd"] = x[m].std(ddof=1)
    rows.append(row)
rows.append({"configuración": "buy_and_hold (ref.)", "total_return_media": bh_ret, "total_return_sd": np.nan, "sharpe_media": bh_sh,
             "sharpe_sd": np.nan, "max_drawdown_media": bh_dd, "max_drawdown_sd": np.nan, "n_trades_media": 2.0, "n_trades_sd": np.nan,
             "llm_cost_media": 0.0, "llm_cost_sd": np.nan})
T = pd.DataFrame(rows); T.to_csv(f"{a.out}/summary_3runs.csv", index=False)
def f(m, s, d=2):
    return f"{m:.{d}f}" if np.isnan(s) else f"{m:.{d}f} ± {s:.{d}f}"
md = ["| Configuración | Retorno total (%) | Sharpe | Max drawdown (%) | Operaciones por activo | Coste LLM (USD, 4 activos) |", "|---|---|---|---|---|---|"]
for _, r in T.iterrows():
    md.append(f"| {r['configuración']} | {f(r.total_return_media, r.total_return_sd)} | {f(r.sharpe_media, r.sharpe_sd)} | "
              f"{f(r.max_drawdown_media, r.max_drawdown_sd)} | {f(r.n_trades_media, r.n_trades_sd, 1)} | {f(r.llm_cost_media, r.llm_cost_sd, 3)} |")
open(f"{a.out}/summary_3runs.md", "w", encoding="utf-8").write("\n".join(md) + "\n\nMedia de 4 activos por ejecución; valor ± desviación típica entre las 3 ejecuciones. buy_and_hold: una sola ejecución (determinista).\n")

# ---- Figura: riesgo-retorno y Sharpe ----
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for s in SETUPS:
    x = per_run[per_run.setup == s]
    ax[0].scatter(x.max_drawdown, x.total_return, color=COL[s], alpha=.45, s=30)
    ax[0].scatter(x.max_drawdown.mean(), x.total_return.mean(), color=COL[s], s=130, edgecolor="k", label=LAB[s], zorder=3)
ax[0].scatter(bh_dd, bh_ret, marker="*", color="k", s=220, label="buy-and-hold", zorder=3)
ax[0].axhline(0, color="k", lw=.6); ax[0].set_xlabel("Máximo drawdown medio de 4 activos (%)"); ax[0].set_ylabel("Retorno total medio de 4 activos (%)")
ax[0].set_title("Retorno y drawdown (puntos claros: ejecuciones; grandes: media)"); ax[0].grid(alpha=.3); ax[0].legend(fontsize=8)
for i, s in enumerate(SETUPS):
    v = per_run[per_run.setup == s].sort_values("run").sharpe.values
    ax[1].bar(i, v.mean(), color=COL[s], alpha=.6); ax[1].scatter([i] * 3, v, color="k", s=18, zorder=3)
ax[1].axhline(bh_sh, color="k", ls="--", lw=1, label=f"buy-and-hold ({bh_sh:.2f})"); ax[1].axhline(0, color="k", lw=.6)
ax[1].set_xticks(range(4)); ax[1].set_xticklabels([LAB[s] for s in SETUPS], fontsize=8); ax[1].set_title("Sharpe medio de 4 activos"); ax[1].legend(fontsize=8); ax[1].grid(axis="y", alpha=.3)
fig.tight_layout(); fig.savefig(f"{a.out}/risk_return.png", dpi=160); plt.close(fig)

# ---- Figuras: posición en acciones en el tiempo, con los episodios ----
def positions(tdir, ticker, episodes, fname):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), sharey=True)
    for ax, r in zip(axes, RUNS):
        for s in SETUPS:
            p = glob.glob(os.path.join(tdir, "**", f"trades_{s}__{ticker}__{r}.csv"), recursive=True)
            if not p: raise SystemExit(f"falta trades_{s}__{ticker}__{r}.csv en {tdir}")
            t = pd.read_csv(p[0]).sort_values(["execution_date", "signal_date"])
            sgn = np.where(t.type == "buy", 1, -1); sh = np.cumsum(sgn * t.quantity.values)
            xs = [pd.Timestamp("2024-08-01")] + list(pd.to_datetime(t.execution_date)) + [pd.Timestamp("2025-01-31")]
            ys = [0] + list(sh) + [sh[-1]]
            ax.step(xs, ys, where="post", color=COL[s], lw=1.6 if s == "full_system" else 1.1, label=LAB[s], alpha=.9)
        for d, lab in episodes:
            ax.axvline(pd.Timestamp(d), color="k", ls="--", lw=.9); ax.text(pd.Timestamp(d), ax.get_ylim()[1] * .97, " " + lab, fontsize=7, va="top")
        ax.set_title(r); ax.grid(alpha=.3); ax.tick_params(axis="x", rotation=30)
    axes[0].set_ylabel(f"Acciones en cartera ({ticker})"); axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle(f"{ticker}: posición en acciones por configuración y ejecución (líneas discontinuas: señal de los episodios)")
    fig.tight_layout(); fig.savefig(f"{a.out}/{fname}", dpi=160); plt.close(fig)
positions(a.trades_nvda, "NVDA", [("2024-08-01", "08-01"), ("2024-09-12", "09-12")], "positions_NVDA.png")
positions(a.trades_xom, "XOM", [("2024-09-26", "09-26")], "positions_XOM.png")
print(T.round(3).to_string(index=False)); print("\n" + "\n".join(md))
