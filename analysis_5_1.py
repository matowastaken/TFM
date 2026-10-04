"""Fase 5.1 - Analisis cuantitativo de la ablacion.
Uso: python analysis_5_1.py --consolidated consolidated_results.csv --days-dir <dir con *.days.jsonl> --out analysis\\5_1
"""
import argparse, json, glob, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--consolidated", default="consolidated_results.csv")
ap.add_argument("--days-dir", default="days_jsonl")
ap.add_argument("--out", default="analysis/5_1")
ap.add_argument("--low-conf", type=float, default=0.5)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

ROLES = {"technical": "no_technical", "sentiment": "no_sentiment", "fundamental": "no_fundamental"}
TICKERS = ["JPM", "JNJ", "XOM", "NVDA"]
df = pd.read_csv(a.consolidated)
df = df[df.setup != "buy_and_hold"].copy()
for c in ["total_return", "annual_return", "max_drawdown"]:
    df[c] = df[c] * 100  # a pp

# ---------- 1. Contribucion marginal por rol y ticker ----------
# Convencion: positivo = el rol aporta.
#  retorno/Sharpe/Calmar: full - sin_rol ; drawdown: sin_rol - full (positivo = el rol reduce el DD)
METRICS = {"total_return": +1, "sharpe": +1, "calmar": +1, "max_drawdown": -1, "n_trades": +1}
full = df[df.setup == "full_system"].set_index("ticker")
rows = []
for role, setup in ROLES.items():
    w = df[df.setup == setup].set_index("ticker")
    for t in TICKERS:
        for m, sgn in METRICS.items():
            rows.append(dict(role=role, ticker=t, metric=m,
                             contribution=sgn * (full.loc[t, m] - w.loc[t, m])))
mc = pd.DataFrame(rows)
mc.to_csv(f"{a.out}/marginal_contribution_by_ticker.csv", index=False)

def summarize(mc):
    out = []
    for (role, m), g in mc.groupby(["role", "metric"]):
        g = g.set_index("ticker").contribution
        out.append(dict(role=role, metric=m, mean4=g.mean(), mean_ex_nvda=g.drop("NVDA").mean(),
                        median4=g.median(), n_pos=int((g > 0).sum()), n_neg=int((g < 0).sum()),
                        **{t: g[t] for t in TICKERS}))
    return pd.DataFrame(out)
sm = summarize(mc)
# n_trades no tiene signo "bueno/malo": se deja como descriptivo
sm.to_csv(f"{a.out}/marginal_contribution_summary.csv", index=False)

# Criterio pre-fijado de "rol perjudicial": contribucion negativa en retorno Y (sharpe o calmar) Y >=3/4 tickers (en retorno)
crit = []
for role in ROLES:
    g = sm[sm.role == role].set_index("metric")
    neg_ret = g.loc["total_return", "mean4"] < 0
    neg_ratio = (g.loc["sharpe", "mean4"] < 0) and (g.loc["calmar", "mean4"] < 0)
    neg_tickers = g.loc["total_return", "n_neg"] >= 3
    pos_tickers = g.loc["total_return", "n_pos"] >= 3
    crit.append(dict(role=role, ret_mean=g.loc["total_return", "mean4"], sharpe_mean=g.loc["sharpe", "mean4"],
                     calmar_mean=g.loc["calmar", "mean4"], n_pos_ret=g.loc["total_return", "n_pos"],
                     perjudicial=bool(neg_ret and neg_ratio and neg_tickers),
                     aporta_consistente=bool(g.loc["total_return", "mean4"] > 0 and g.loc["sharpe", "mean4"] > 0
                                             and g.loc["calmar", "mean4"] > 0 and pos_tickers)))
crit = pd.DataFrame(crit); crit.to_csv(f"{a.out}/criterio_rol_perjudicial.csv", index=False)

# Sensibilidad: leave-one-ticker-out sobre el retorno medio
loo = []
for role in ROLES:
    g = mc[(mc.role == role) & (mc.metric == "total_return")].set_index("ticker").contribution
    for t in TICKERS:
        loo.append(dict(role=role, dropped=t, mean_rest=g.drop(t).mean()))
pd.DataFrame(loo).pivot(index="role", columns="dropped", values="mean_rest").to_csv(f"{a.out}/leave_one_ticker_out.csv")

# Efecto conjunto: full vs baseline
bl = df[df.setup == "baseline_price_only"].set_index("ticker")
conj = pd.DataFrame({m: (full[m] - bl[m]) for m in ["total_return", "max_drawdown", "sharpe", "n_trades"]})
conj.to_csv(f"{a.out}/full_vs_baseline.csv")

# ---------- 2. Diagnostico de roles desde days.jsonl ----------
recs, trd = [], []
for f in glob.glob(os.path.join(a.days_dir, "*__*.days.jsonl")):
    setup, tk = os.path.basename(f).replace(".days.jsonl", "").split("__")
    for l in open(f, encoding="utf-8"):
        d = json.loads(l)
        if d.get("status") != "ok": continue
        for r in d.get("analyst_reports") or []:
            recs.append(dict(setup=setup, ticker=tk, date=d["date"], role=r["role"], signal=r["signal"],
                             score=r["score"], confidence=r["confidence"]))
        t = d.get("trader_output") or {}
        trd.append(dict(setup=setup, ticker=tk, date=d["date"], pos=d.get("position_state"),
                        action=t.get("action"), size=t.get("size"), conf=t.get("confidence")))
R = pd.DataFrame(recs); T = pd.DataFrame(trd)
R.to_csv(f"{a.out}/_analyst_reports_flat.csv", index=False)

def diag(g):
    return pd.Series(dict(n=len(g), pct_bullish=(g.signal == "bullish").mean() * 100,
                          pct_neutral=(g.signal == "neutral").mean() * 100,
                          pct_bearish=(g.signal == "bearish").mean() * 100,
                          mean_score=g.score.mean(), mean_abs_score=g.score.abs().mean(),
                          mean_conf=g.confidence.mean(),
                          pct_neutral_lowconf=((g.signal == "neutral") & (g.confidence < a.low_conf)).mean() * 100,
                          pct_lowconf=(g.confidence < a.low_conf).mean() * 100))
d_role = R.groupby("role").apply(diag, include_groups=False); d_role.to_csv(f"{a.out}/diag_roles_global.csv")
d_rt = R.groupby(["role", "ticker"]).apply(diag, include_groups=False); d_rt.to_csv(f"{a.out}/diag_roles_by_ticker.csv")

# Ruido de los analistas "gratis": mismo rol/ticker/fecha en configuraciones distintas = mismo prompt, mismo input
# (el input de un analista no depende de la configuracion), asi que las discrepancias son puro muestreo del LLM.
nz = []
for (role, tk, dt), g in R.groupby(["role", "ticker", "date"]):
    if len(g) < 2: continue
    nz.append(dict(role=role, ticker=tk, date=dt, k=len(g), same_signal=g.signal.nunique() == 1,
                   score_range=g.score.max() - g.score.min(), score_std=g.score.std(ddof=0),
                   sign_flip=(g.score.max() > 0.1) and (g.score.min() < -0.1)))
N = pd.DataFrame(nz)
noise = N.groupby("role").agg(n_cells=("k", "size"), pct_same_signal=("same_signal", lambda s: s.mean() * 100),
                              mean_score_std=("score_std", "mean"), mean_score_range=("score_range", "mean"),
                              pct_sign_flip=("sign_flip", lambda s: s.mean() * 100))
noise.to_csv(f"{a.out}/noise_analysts_same_input.csv")
N.groupby(["role", "ticker"]).agg(pct_same_signal=("same_signal", lambda s: s.mean() * 100),
                                  mean_score_std=("score_std", "mean")).to_csv(f"{a.out}/noise_analysts_by_ticker.csv")

# Acuerdo entre roles (mismo dia, misma config full): correlacion de scores
F = R[R.setup == "full_system"].pivot_table(index=["ticker", "date"], columns="role", values="score")
F.corr().to_csv(f"{a.out}/role_score_correlation_full.csv")
F.groupby("ticker").apply(lambda g: g.corr().stack(), include_groups=False).to_csv(f"{a.out}/role_score_correlation_by_ticker.csv")

# ---------- 3. Decisiones del trader ----------
T["size"] = T["size"].fillna(0)
def tdiag(g):
    b = g[g.action == "buy"]; s = g[g.action == "sell"]
    return pd.Series(dict(n=len(g), pct_buy=(g.action == "buy").mean() * 100, pct_hold=(g.action == "hold").mean() * 100,
                          pct_sell=(g.action == "sell").mean() * 100, mean_size_buy=b["size"].mean(),
                          mean_size_sell=s["size"].mean(), mean_conf=g.conf.mean()))
T.groupby("setup").apply(tdiag, include_groups=False).to_csv(f"{a.out}/trader_decisions_by_setup.csv")
T.groupby(["setup", "ticker"]).apply(tdiag, include_groups=False).to_csv(f"{a.out}/trader_decisions_by_setup_ticker.csv")

# ---------- 4. Figura ----------
order = ["technical", "sentiment", "fundamental"]; lab = {"technical": "Técnico", "sentiment": "Sentimiento", "fundamental": "Fundamental"}
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
cols = ["JPM", "JNJ", "XOM", "NVDA", "Media 4", "Media sin NVDA"]
titles = {"total_return": "Retorno total (pp)", "sharpe": "Sharpe", "max_drawdown": "Reducción de max drawdown (pp)"}
for ax, m in zip(axes, titles):
    s = sm[sm.metric == m].set_index("role").loc[order]
    x = np.arange(len(order)); w = 0.13
    for i, c in enumerate(["JPM", "JNJ", "XOM", "NVDA"]):
        ax.bar(x + (i - 2.5) * w, s[c], w, label=c, alpha=.8)
    ax.bar(x + 1.5 * w, s["mean4"], w, label="Media 4", color="black")
    ax.bar(x + 2.5 * w, s["mean_ex_nvda"], w, label="Media sin NVDA", color="grey", hatch="//")
    ax.axhline(0, color="k", lw=.8); ax.set_xticks(x); ax.set_xticklabels([lab[r] for r in order])
    ax.set_title(titles[m]); ax.grid(axis="y", alpha=.3)
axes[0].legend(fontsize=7, ncol=2)
fig.suptitle("Contribución marginal por rol (positivo = el rol aporta)")
fig.tight_layout(); fig.savefig(f"{a.out}/marginal_contribution_bars.png", dpi=160)

pd.set_option("display.width", 220, "display.max_columns", 30, "display.float_format", lambda v: f"{v:,.2f}")
for name, obj in [("CONTRIB. MARGINAL (resumen)", sm), ("CRITERIO", crit), ("LEAVE-ONE-TICKER-OUT (ret pp)", pd.read_csv(f"{a.out}/leave_one_ticker_out.csv")),
                  ("FULL - BASELINE", conj), ("DIAG ROLES", d_role), ("DIAG ROLES x TICKER", d_rt),
                  ("RUIDO ANALISTAS (mismo input)", noise), ("CORR SCORES (full)", F.corr()),
                  ("TRADER x SETUP", T.groupby('setup').apply(tdiag, include_groups=False))]:
    print("\n=== " + name + " ===\n", obj.to_string())
