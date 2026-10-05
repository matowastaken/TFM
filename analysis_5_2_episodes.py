"""Fase 5.2 - selección de episodios y dossier cualitativo a partir de los days.jsonl de las 3 ejecuciones.
Uso (PowerShell, una línea o con acento grave):
  python analysis_5_2_episodes.py --days-dirs C:\\finsaber-data\\results\\_run_logs C:\\finsaber-data\\results_rep2\\_run_logs C:\\finsaber-data\\results_rep3\\_run_logs --out analysis\\5_2
Criterio de selección (fijado antes de leer los textos): un día es 'dependiente de rol' si, en una misma ejecución,
full_system y la configuración sin ese rol parten de la MISMA posición y toman acciones distintas. Los candidatos a episodio
son los días que cumplen eso en las 3 ejecuciones (efecto del rol y no ruido del trader).
Todo sale de los ficheros; no se usan precios (los days.jsonl no los contienen).
"""
import argparse, glob, json, os
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--days-dirs", nargs=3, required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
RUNS = ["rep1", "rep2", "rep3"]; TICKERS = ["JPM", "JNJ", "XOM", "NVDA"]
SETUPS = ["full_system", "no_technical", "no_sentiment", "no_fundamental"]
W = {"technical": "no_technical", "sentiment": "no_sentiment", "fundamental": "no_fundamental"}

recs = {}
for r, d in zip(RUNS, a.days_dirs):
    for f in glob.glob(os.path.join(d, "*__*.days.jsonl")):
        s, t = os.path.basename(f).replace(".days.jsonl", "").split("__")
        if s not in SETUPS: continue
        for l in open(f, encoding="utf-8"):
            if not l.strip(): continue
            x = json.loads(l)
            if x.get("status") != "ok": continue
            to = x.get("trader_output") or {}
            recs[(r, s, t, x["date"])] = dict(run=r, setup=s, ticker=t, date=x["date"], pos=x["position_state"],
                action=to.get("action"), size=to.get("size") or 0.0, trader_rationale=to.get("rationale", ""),
                reports={y["role"]: y for y in x.get("analyst_reports") or []})
df = pd.DataFrame(recs.values())

# a) coherencia lógica de las órdenes
b = df[df.action == "buy"]; sl = df[df.action == "sell"]
chk = pd.DataFrame({"compras": b.groupby("setup").size(), "compras_estando_en_posicion": b[b.pos.str.startswith("holding")].groupby("setup").size(),
                    "ventas": sl.groupby("setup").size(), "ventas_sin_posicion": sl[sl.pos.str.startswith("no position")].groupby("setup").size()}).fillna(0).astype(int)
chk.to_csv(f"{a.out}/order_logic_check.csv")

# b) operaciones por configuración con consenso entre ejecuciones
nh = df[df.action != "hold"]
cons = nh.groupby(["ticker", "setup", "date", "action"]).agg(n_runs=("run", "nunique"),
        sizes=("size", lambda x: sorted(set(x)))).reset_index().sort_values(["ticker", "date", "setup"])
cons.to_csv(f"{a.out}/trades_by_config_all_runs.csv", index=False)

# c) días dependientes de rol
rows = []
for role, w in W.items():
    for t in TICKERS:
        for r in RUNS:
            for (rr, s, tt, d), f in recs.items():
                if rr != r or s != "full_system" or tt != t: continue
                x = recs.get((r, w, t, d))
                if x and f["pos"] == x["pos"] and f["action"] != x["action"]:
                    rows.append(dict(role=role, ticker=t, date=d, run=r, pos=f["pos"], full_action=f["action"], full_size=f["size"],
                                     abl_action=x["action"], abl_size=x["size"]))
RD = pd.DataFrame(rows)
RD.to_csv(f"{a.out}/role_dependent_days.csv", index=False)
cnt = RD.groupby(["role", "ticker", "date"]).run.nunique().reset_index(name="n_runs")
summ = cnt.groupby(["role", "ticker"]).agg(dias_en_alguna=("date", "size"), dias_en_2_o_mas=("n_runs", lambda s: int((s >= 2).sum())),
                                           dias_en_las_3=("n_runs", lambda s: int((s == 3).sum()))).reset_index()
summ.to_csv(f"{a.out}/role_dependent_summary.csv", index=False)
cand = cnt[cnt.n_runs == 3].sort_values(["ticker", "date", "role"]); cand.to_csv(f"{a.out}/episode_candidates_3of3.csv", index=False)

# d) dossier de texto de los candidatos (3 de 3)
L = ["# Dossier de episodios candidatos (dependientes de rol en las 3 ejecuciones)\n",
     "Los textos son salidas del LLM tal cual (en inglés). No contienen precios; el efecto en el retorno se evalúa con trades.csv.\n"]
for _, c in cand.iterrows():
    w = W[c.role]
    L.append(f"\n## {c.ticker} · {c.date} · rol: {c.role}\n")
    for r in RUNS:
        f = recs[(r, "full_system", c.ticker, c.date)]; x = recs[(r, w, c.ticker, c.date)]
        L.append(f"- **{r}** · posición: {f['pos']} · full_system: {f['action']} (size {f['size']}) · {w}: {x['action']} (size {x['size']})")
    r0 = recs[("rep1", "full_system", c.ticker, c.date)]
    L.append("\nInformes de analista en full_system (rep1):")
    for role in ["technical", "sentiment", "fundamental"]:
        y = r0["reports"].get(role)
        if y: L.append(f"- {role}: {y['signal']} (score {y['score']}, conf {y['confidence']}) — {y['rationale']}")
    for r in RUNS:
        f = recs[(r, "full_system", c.ticker, c.date)]; x = recs[(r, w, c.ticker, c.date)]
        L.append(f"\n{r} · trader full_system: {f['trader_rationale']}")
        L.append(f"{r} · trader {w}: {x['trader_rationale']}")
open(f"{a.out}/episodes_dossier.md", "w", encoding="utf-8").write("\n".join(L))

pd.set_option("display.width", 220)
print("== coherencia de órdenes ==\n", chk.to_string())
print("\n== días dependientes de rol ==\n", summ.to_string(index=False))
print("\n== candidatos 3/3 ==\n", cand.to_string(index=False))
print("\nDossier:", f"{a.out}/episodes_dossier.md")
