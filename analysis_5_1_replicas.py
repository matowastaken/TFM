"""Fase 5.1 (réplicas) - dispersión entre ejecuciones y contribución marginal con 3 ejecuciones.
Uso:
  python analysis_5_1_replicas.py ^
    --consolidated analysis\\replicas\\rep1\\consolidated_results.csv analysis\\replicas\\rep2\\consolidated_results.csv analysis\\replicas\\rep3\\consolidated_results.csv ^
    --summaries <rep1 summary> <rep2 summary> <rep3 summary> ^
    --days-dirs C:\\finsaber-data\\results\\_run_logs C:\\finsaber-data\\results_rep2\\_run_logs C:\\finsaber-data\\results_rep3\\_run_logs ^
    --out analysis\\5_1_replicas
Las tres listas van en el orden rep1 rep2 rep3. Todo se calcula desde los ficheros; no hay cifras externas.
"""
import argparse, glob, json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--consolidated", nargs=3, required=True)
ap.add_argument("--summaries", nargs=3, required=True)
ap.add_argument("--days-dirs", nargs=3, required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

RUNS = ["rep1", "rep2", "rep3"]
SETUPS = ["full_system", "no_technical", "no_sentiment", "no_fundamental"]
TICKERS = ["JPM", "JNJ", "XOM", "NVDA"]
ROLES = {"technical": "no_technical", "sentiment": "no_sentiment", "fundamental": "no_fundamental"}
ROLE_ES = {"technical": "Técnico", "sentiment": "Sentimiento", "fundamental": "Fundamental"}
SIGN = {"total_return": +1, "sharpe": +1, "calmar": +1, "max_drawdown": -1}  # contrib = SIGN*(full - sin_rol)
pd.set_option("display.width", 250, "display.max_columns", 40, "display.float_format", lambda v: f"{v:,.2f}")

# ---------------- carga ----------------
frames = []
for r, p in zip(RUNS, a.consolidated):
    d = pd.read_csv(p)
    d["run"] = r
    frames.append(d)
allc = pd.concat(frames, ignore_index=True)
bh = allc[(allc.setup == "buy_and_hold")].copy()
D = allc[allc.setup.isin(SETUPS)].copy()
D["total_return"] *= 100; D["max_drawdown"] *= 100; D["annual_return"] *= 100
assert len(D) == 48, len(D)
D.to_csv(f"{a.out}/_all_runs_long.csv", index=False)

# ---------------- A. QA de las tres ejecuciones ----------------
qa = []
for r, p in zip(RUNS, a.summaries):
    s = pd.read_csv(p)
    s_llm = s[s.setup.isin(SETUPS)]
    qa.append(dict(run=r, pares_en_summary=len(s), pares_comparables=len(s_llm), attempts_max=s.attempts.max(),
                   dias_min=s.days_evaluated.min(), dias_max=s.days_evaluated.max(),
                   fallos_dia=int(s.day_failures.sum()), ordenes_rechazadas=int(s.rejected_orders.sum()),
                   coste_usd_comparables=s_llm.cost_usd.sum(), coste_usd_total=s.cost_usd.sum(),
                   minutos_reloj=s.wall_seconds.sum() / 60))
qa = pd.DataFrame(qa); qa.to_csv(f"{a.out}/qa_runs.csv", index=False)

# control cruzado: retorno de rep1 recalculado vs summary de rep1 (misma ejecución)
s1 = pd.read_csv(a.summaries[0]).set_index(["setup", "ticker"]).total_return * 100
c1 = D[D.run == "rep1"].set_index(["setup", "ticker"]).total_return
xchk = (c1 - s1.reindex(c1.index)).abs().max()

# ---------------- B. dispersión por celda ----------------
M = ["total_return", "sharpe", "max_drawdown", "calmar", "n_trades"]
g = D.groupby(["setup", "ticker"])
cells = []
for (s, t), x in g:
    row = dict(setup=s, ticker=t, n_runs=len(x))
    for m in M:
        v = x[m].astype(float)
        row.update({f"{m}_mean": v.mean(), f"{m}_sd": v.std(ddof=1), f"{m}_min": v.min(), f"{m}_max": v.max(),
                    f"{m}_range": v.max() - v.min()})
    row.update({f"ret_{r}": x[x.run == r].total_return.iloc[0] for r in RUNS})
    cells.append(row)
cells = pd.DataFrame(cells)
cells["setup"] = pd.Categorical(cells.setup, SETUPS); cells["ticker"] = pd.Categorical(cells.ticker, TICKERS)
cells = cells.sort_values(["ticker", "setup"]); cells.to_csv(f"{a.out}/cell_dispersion.csv", index=False)

def pooled(sdcol, by=None):
    f = lambda x: np.sqrt((x ** 2).mean())
    return cells[sdcol].groupby(cells[by]).apply(f, include_groups=False) if by else f(cells[sdcol])
noise = pd.DataFrame({m: [pooled(f"{m}_sd")] for m in M}, index=["sd_agrupada_16_celdas"])
noise_cfg = pd.DataFrame({m: pooled(f"{m}_sd", "setup") for m in M})
noise_tk = pd.DataFrame({m: pooled(f"{m}_sd", "ticker") for m in M})
noise.to_csv(f"{a.out}/noise_pooled.csv"); noise_cfg.to_csv(f"{a.out}/noise_by_config.csv"); noise_tk.to_csv(f"{a.out}/noise_by_ticker.csv")
rng_ret = cells.total_return_range

# media de 4 tickers por config y ejecución -> ruido en el agregado
agg = D.groupby(["run", "setup"])[["total_return", "sharpe", "max_drawdown", "calmar", "n_trades"]].mean().reset_index()
agg_stats = agg.groupby("setup")[["total_return", "sharpe", "max_drawdown", "calmar"]].agg(["mean", "std", "min", "max"])
agg_stats = agg_stats.reindex(SETUPS)
agg.to_csv(f"{a.out}/mean4_by_run_config.csv", index=False); agg_stats.to_csv(f"{a.out}/mean4_by_config_stats.csv")

# ---------------- C. contribución marginal por ejecución ----------------
rows = []
for r in RUNS:
    for role, w in ROLES.items():
        for t in TICKERS:
            f = D[(D.run == r) & (D.setup == "full_system") & (D.ticker == t)].iloc[0]
            x = D[(D.run == r) & (D.setup == w) & (D.ticker == t)].iloc[0]
            for m, sg in SIGN.items():
                rows.append(dict(run=r, role=role, ticker=t, metric=m, contribution=sg * (f[m] - x[m])))
MC = pd.DataFrame(rows); MC.to_csv(f"{a.out}/marginal_contribution_by_run_ticker.csv", index=False)

# C1. por ejecución, media de 4 tickers (y sin NVDA)
pr = []
for (r, role, m), x in MC.groupby(["run", "role", "metric"]):
    x = x.set_index("ticker").contribution
    pr.append(dict(run=r, role=role, metric=m, mean4=x.mean(), mean_ex_nvda=x.drop("NVDA").mean(), n_pos_tickers=int((x > 0).sum())))
PR = pd.DataFrame(pr); PR.to_csv(f"{a.out}/marginal_by_run_mean4.csv", index=False)
stab = []
for (role, m), x in PR.groupby(["role", "metric"]):
    stab.append(dict(role=role, metric=m, mean_of_runs=x.mean4.mean(), sd_runs=x.mean4.std(ddof=1), min_run=x.mean4.min(),
                     max_run=x.mean4.max(), runs_positive=int((x.mean4 > 0).sum()),
                     **{f"{r}": x[x.run == r].mean4.iloc[0] for r in RUNS},
                     mean_ex_nvda_of_runs=x.mean_ex_nvda.mean(), runs_positive_ex_nvda=int((x.mean_ex_nvda > 0).sum())))
STAB = pd.DataFrame(stab); STAB.to_csv(f"{a.out}/marginal_stability_mean4.csv", index=False)

# C2. por ticker: media de 3 ejecuciones, sd entre ejecuciones, ejecuciones con signo +
pt = []
for (role, t, m), x in MC.groupby(["role", "ticker", "metric"]):
    pt.append(dict(role=role, ticker=t, metric=m, mean3=x.contribution.mean(), sd3=x.contribution.std(ddof=1),
                   runs_positive=int((x.contribution > 0).sum()),
                   snr=abs(x.contribution.mean()) / x.contribution.std(ddof=1) if x.contribution.std(ddof=1) > 0 else np.nan))
PT = pd.DataFrame(pt); PT.to_csv(f"{a.out}/marginal_by_ticker_3runs.csv", index=False)

# D. contribución marginal sobre la media de 3 ejecuciones (por celda) y criterio pre-fijado
mean3 = D.groupby(["setup", "ticker"])[["total_return", "sharpe", "calmar", "max_drawdown"]].mean()
sm = []
for role, w in ROLES.items():
    for m, sg in SIGN.items():
        v = pd.Series({t: sg * (mean3.loc[("full_system", t), m] - mean3.loc[(w, t), m]) for t in TICKERS})
        sm.append(dict(role=role, metric=m, mean4=v.mean(), mean_ex_nvda=v.drop("NVDA").mean(), n_pos=int((v > 0).sum()),
                       n_neg=int((v < 0).sum()), **v.to_dict()))
SM = pd.DataFrame(sm); SM.to_csv(f"{a.out}/marginal_summary_mean3runs.csv", index=False)
crit = []
for role in ROLES:
    x = SM[SM.role == role].set_index("metric")
    crit.append(dict(role=role, ret=x.loc["total_return", "mean4"], sharpe=x.loc["sharpe", "mean4"], calmar=x.loc["calmar", "mean4"],
                     tickers_pos_ret=x.loc["total_return", "n_pos"],
                     perjudicial=bool(x.loc["total_return", "mean4"] < 0 and x.loc["sharpe", "mean4"] < 0 and x.loc["calmar", "mean4"] < 0
                                      and x.loc["total_return", "n_neg"] >= 3),
                     aporta_consistente=bool(x.loc["total_return", "mean4"] > 0 and x.loc["sharpe", "mean4"] > 0 and x.loc["calmar", "mean4"] > 0
                                             and x.loc["total_return", "n_pos"] >= 3)))
CR = pd.DataFrame(crit); CR.to_csv(f"{a.out}/criterio_rol_perjudicial_3runs.csv", index=False)

# D2. rep1 sola vs media de 3: ¿cambia el signo del agregado de retorno?
cmp_ = []
for role in ROLES:
    r1 = PR[(PR.run == "rep1") & (PR.role == role) & (PR.metric == "total_return")].mean4.iloc[0]
    cmp_.append(dict(role=role, rep1_mean4_ret=r1, media_3_runs_ret=SM[(SM.role == role) & (SM.metric == "total_return")].mean4.iloc[0]))
pd.DataFrame(cmp_).to_csv(f"{a.out}/rep1_vs_3runs_return.csv", index=False)

# ---------------- E. días: ¿de dónde viene el ruido? ----------------
def load_days(dirpath, run):
    R, T = [], []
    for f in glob.glob(os.path.join(dirpath, "*__*.days.jsonl")):
        setup, tk = os.path.basename(f).replace(".days.jsonl", "").split("__")
        if setup not in SETUPS: continue
        for l in open(f, encoding="utf-8"):
            if not l.strip(): continue
            d = json.loads(l)
            if d.get("status") != "ok": continue
            for x in d.get("analyst_reports") or []:
                R.append(dict(run=run, setup=setup, ticker=tk, date=d["date"], role=x["role"], signal=x["signal"], score=x["score"]))
            t = d.get("trader_output") or {}
            T.append(dict(run=run, setup=setup, ticker=tk, date=d["date"], pos=d.get("position_state"), action=t.get("action"), size=t.get("size")))
    return pd.DataFrame(R), pd.DataFrame(T)
Rs, Ts = [], []
for r, p in zip(RUNS, a.days_dirs):
    x, y = load_days(p, r); Rs.append(x); Ts.append(y)
R = pd.concat(Rs); T = pd.concat(Ts)
# analistas: mismo rol/ticker/fecha en todas las configuraciones y ejecuciones (mismo input)
an = []
for (role, tk, dt), x in R.groupby(["role", "ticker", "date"]):
    an.append(dict(role=role, ticker=tk, k=len(x), same_signal=x.signal.nunique() == 1, score_sd=x.score.std(ddof=0)))
an = pd.DataFrame(an)
an_s = an.groupby("role").agg(celdas=("k", "size"), muestras_por_celda=("k", "mean"), pct_misma_senal=("same_signal", lambda s: s.mean() * 100),
                              sd_score_media=("score_sd", "mean"))
an_s.to_csv(f"{a.out}/analyst_noise_across_runs.csv")
# trader: misma (config, ticker, fecha) en 3 ejecuciones
tr = []
for (s, tk, dt), x in T.groupby(["setup", "ticker", "date"]):
    if len(x) < 3: continue
    same_pos = x.pos.nunique() == 1
    tr.append(dict(setup=s, ticker=tk, date=dt, same_pos=same_pos, same_action=x.action.nunique() == 1,
                   n_buy=int((x.action == "buy").sum()), n_sell=int((x.action == "sell").sum())))
tr = pd.DataFrame(tr)
t_all = dict(dias_celda=len(tr), pct_misma_posicion_3runs=tr.same_pos.mean() * 100, pct_misma_accion_3runs=tr.same_action.mean() * 100)
sp = tr[tr.same_pos]
t_all["pct_misma_accion_dado_misma_posicion"] = sp.same_action.mean() * 100
t_all["dias_misma_posicion"] = len(sp)
pd.Series(t_all).to_csv(f"{a.out}/trader_agreement_across_runs.csv", header=["valor"])
tr_cell = tr.groupby(["setup", "ticker"]).agg(pct_misma_posicion=("same_pos", lambda s: s.mean() * 100),
                                             pct_misma_accion=("same_action", lambda s: s.mean() * 100))
tr_cell.to_csv(f"{a.out}/trader_agreement_by_cell.csv")

# ---------------- F. figuras ----------------
cols = {"full_system": "#1f77b4", "no_technical": "#d62728", "no_sentiment": "#2ca02c", "no_fundamental": "#ff7f0e"}
fig, axes = plt.subplots(1, 4, figsize=(16, 4.3), sharey=False)
for ax, t in zip(axes, TICKERS):
    for i, s in enumerate(SETUPS):
        v = D[(D.ticker == t) & (D.setup == s)].sort_values("run").total_return.values
        ax.scatter([i] * len(v), v, color=cols[s], s=38, zorder=3)
        ax.hlines(v.mean(), i - .25, i + .25, color="k", lw=1.5, zorder=4)
    b = bh[bh.ticker == t].total_return.iloc[0] * 100
    ax.axhline(b, color="grey", ls="--", lw=1, label=f"buy-and-hold ({b:.1f} %)"); ax.axhline(0, color="k", lw=.6)
    ax.set_xticks(range(4)); ax.set_xticklabels(["full", "sin téc.", "sin sent.", "sin fund."], fontsize=8)
    ax.set_title(t); ax.grid(axis="y", alpha=.3); ax.legend(fontsize=7, loc="best")
axes[0].set_ylabel("Retorno total (%)")
fig.suptitle("Retorno total por celda: 3 ejecuciones (puntos) y media (línea negra)"); fig.tight_layout()
fig.savefig(f"{a.out}/dispersion_return_by_cell.png", dpi=160); plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
for ax, (role, _) in zip(axes, ROLES.items()):
    for i, t in enumerate(TICKERS + ["Media 4"]):
        if t == "Media 4":
            v = PR[(PR.role == role) & (PR.metric == "total_return")].sort_values("run").mean4.values
        else:
            v = MC[(MC.role == role) & (MC.metric == "total_return") & (MC.ticker == t)].sort_values("run").contribution.values
        ax.scatter([i] * 3, v, color="k" if t == "Media 4" else "#1f77b4", s=36, zorder=3)
        ax.hlines(v.mean(), i - .25, i + .25, color="r", lw=1.5, zorder=4)
    ax.axhline(0, color="k", lw=.6); ax.set_xticks(range(5)); ax.set_xticklabels(TICKERS + ["Media 4"], fontsize=8)
    ax.set_title(ROLE_ES[role]); ax.grid(axis="y", alpha=.3)
axes[0].set_ylabel("Contribución marginal al retorno (pp)")
fig.suptitle("Contribución marginal por ejecución (puntos) y media de 3 (línea roja); positivo = el rol aporta"); fig.tight_layout()
fig.savefig(f"{a.out}/marginal_by_run.png", dpi=160); plt.close(fig)

# ---------------- salida por consola ----------------
def show(t, o): print(f"\n=== {t} ===\n{o.to_string() if hasattr(o,'to_string') else o}")
show("A. QA por ejecución", qa.T)
print(f"Control rep1: max|retorno consolidado - summary| (pp) = {xchk:.3g}")
show("B. SD agrupada entre ejecuciones (16 celdas)", noise)
show("B. SD agrupada por configuración", noise_cfg)
show("B. SD agrupada por ticker", noise_tk)
show("B. Rango del retorno entre ejecuciones por celda (pp): resumen", rng_ret.describe())
show("B. Celdas", cells[["ticker", "setup", "ret_rep1", "ret_rep2", "ret_rep3", "total_return_mean", "total_return_sd", "total_return_range", "sharpe_sd", "max_drawdown_sd", "n_trades_mean"]])
show("B. Media de 4 tickers por config y ejecución (retorno total)", agg.pivot(index="setup", columns="run", values="total_return").reindex(SETUPS))
show("B. Estadísticos del agregado (media 4 tickers)", agg_stats)
show("C. Contribución marginal por ejecución, media 4 tickers (retorno total)", PR[PR.metric == "total_return"].pivot(index="role", columns="run", values="mean4"))
show("C. Estabilidad (media 4 tickers)", STAB[["role", "metric", "mean_of_runs", "sd_runs", "min_run", "max_run", "runs_positive", "mean_ex_nvda_of_runs", "runs_positive_ex_nvda"]])
show("C2. Por ticker, retorno: media 3 / sd / ejecuciones +", PT[PT.metric == "total_return"].drop(columns="metric"))
show("D. Resumen contribución marginal (media de 3 ejecuciones por celda)", SM)
show("D. Criterio pre-fijado", CR)
show("D2. rep1 sola vs media 3", pd.read_csv(f"{a.out}/rep1_vs_3runs_return.csv"))
show("E. Ruido de analistas entre ejecuciones y configs", an_s)
show("E. Acuerdo del trader entre las 3 ejecuciones", pd.Series(t_all))
show("E. Acuerdo del trader por celda", tr_cell)
show("buy-and-hold rep1 (retorno total %)", bh[bh.run == "rep1"].set_index("ticker").total_return * 100)
