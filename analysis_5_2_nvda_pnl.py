"""Fase 5.2 - P&L por ciclo de posición en NVDA, a partir de los trades.csv de las 3 ejecuciones.
Uso:
  python analysis_5_2_nvda_pnl.py --trades-dir analysis\\trades_nvda --consolidated analysis\\replicas\\rep1\\consolidated_results.csv analysis\\replicas\\rep2\\consolidated_results.csv analysis\\replicas\\rep3\\consolidated_results.csv --out analysis\\5_2
Un ciclo = periodo entre dos momentos con 0 acciones (compra inicial -> venta(s) hasta quedar a 0).
P&L del ciclo = ingresos de las ventas - coste de las compras - comisiones, con los precios y comisiones de trades.csv.
Se valida contra final_value del consolidado (capital inicial 100.000 USD).
"""
import argparse, glob, os, re
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--trades-dir", required=True)
ap.add_argument("--consolidated", nargs=3, required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
CAP = 100000.0
RUNS = ["rep1", "rep2", "rep3"]; SETUPS = ["full_system", "no_technical", "no_sentiment", "no_fundamental"]

cycles, ops = [], []
for f in sorted(glob.glob(os.path.join(a.trades_dir, "trades_*__NVDA__rep*.csv"))):
    m = re.match(r"trades_(.+)__NVDA__(rep\d)\.csv", os.path.basename(f))
    s, r = m.group(1), m.group(2)
    t = pd.read_csv(f).sort_values(["execution_date", "signal_date"]).reset_index(drop=True)
    pos = 0; cur = None
    for _, x in t.iterrows():
        q = int(x.quantity); sgn = 1 if x.type == "buy" else -1
        cash = -sgn * q * x.price - x.commission
        ops.append(dict(setup=s, run=r, signal_date=x.signal_date, execution_date=x.execution_date, type=x.type,
                        price=x.price, quantity=q, commission=x.commission))
        if pos == 0:
            cur = dict(setup=s, run=r, entry_signal=x.signal_date, entry_exec=x.execution_date, entry_price=x.price,
                       shares_bought=0, n_sells=0, pnl=0.0)
        cur["pnl"] += cash
        if x.type == "buy": cur["shares_bought"] += q
        else: cur["n_sells"] += 1
        pos += sgn * q
        if pos == 0:
            cur["exit_exec"] = x.execution_date; cur["exit_signal"] = x.signal_date
            cycles.append(cur); cur = None
    if pos != 0:
        raise SystemExit(f"{s}/{r}: posición abierta al final ({pos} acciones); revisar trades.csv")
C = pd.DataFrame(cycles); O = pd.DataFrame(ops)
C["pnl_pct_capital"] = C.pnl / CAP * 100
C["order"] = C.setup.map({s: i for i, s in enumerate(SETUPS)})
C = C.sort_values(["order", "run", "entry_exec"]).drop(columns="order")
C.to_csv(f"{a.out}/nvda_cycles.csv", index=False); O.to_csv(f"{a.out}/nvda_operations.csv", index=False)

# validación contra el consolidado
cons = pd.concat([pd.read_csv(p).assign(run=r) for r, p in zip(RUNS, a.consolidated)])
cons = cons[(cons.ticker == "NVDA") & cons.setup.isin(SETUPS)].set_index(["setup", "run"])
tot = C.groupby(["setup", "run"]).agg(pnl=("pnl", "sum"), ciclos=("pnl", "size"))
tot["retorno_trades_pct"] = tot.pnl / CAP * 100
tot["retorno_consolidado_pct"] = cons.total_return.reindex(tot.index) * 100
tot["diferencia_pp"] = tot.retorno_trades_pct - tot.retorno_consolidado_pct
tot = tot.reindex(pd.MultiIndex.from_product([SETUPS, RUNS], names=["setup", "run"]))
tot.to_csv(f"{a.out}/nvda_pnl_validation.csv")

# descomposición exacta de (configuración - full_system) por cubeta de entrada (suma = diferencia total de P&L)
BUCKET = {"2024-08-01": "Ago-01", "2024-08-14": "Ago-14", "2024-09-12": "Sep (12/16)", "2024-09-16": "Sep (12/16)",
          "2024-12-11": "Dic-11", "2025-01-15": "Ene-15"}
C["entry_signal"] = C.entry_signal.astype(str).str[:10]
unk = set(C.entry_signal) - set(BUCKET)
if unk: raise SystemExit(f"fechas de entrada sin cubeta definida: {unk}")
C["bucket"] = C.entry_signal.map(BUCKET)
BUCKETS = ["Ago-01", "Ago-14", "Sep (12/16)", "Dic-11", "Ene-15"]
pv = C.pivot_table(index=["setup", "run"], columns="bucket", values="pnl", aggfunc="sum").reindex(columns=BUCKETS).fillna(0.0)
dec = []
for s_ in SETUPS[1:]:
    for r in RUNS:
        d = (pv.loc[(s_, r)] - pv.loc[("full_system", r)]) / CAP * 100  # configuración - full, en pp del capital
        dec.append(dict(setup=s_, run=r, **d.to_dict(), total=d.sum()))
DEC = pd.DataFrame(dec); DEC.to_csv(f"{a.out}/nvda_decomposition_vs_full.csv", index=False)
C.to_csv(f"{a.out}/nvda_cycles.csv", index=False)
pv_pct = pv / CAP * 100; pv_pct["total"] = pv_pct.sum(axis=1); pv_pct.to_csv(f"{a.out}/nvda_pnl_by_bucket.csv")

pd.set_option("display.width", 230, "display.max_columns", 30, "display.float_format", lambda v: f"{v:,.2f}")
print("== Validación: P&L de trades.csv vs retorno consolidado ==\n", tot.to_string())
print("\n== Ciclos de posición (P&L en USD y % del capital inicial) ==")
for s in SETUPS:
    for r in RUNS:
        x = C[(C.setup == s) & (C.run == r)]
        print(f"\n{s} / {r}")
        print(x[["entry_signal", "entry_exec", "exit_exec", "shares_bought", "n_sells", "pnl", "pnl_pct_capital"]].to_string(index=False))

print("\n== P&L por cubeta de entrada (% del capital) ==\n", pv_pct.reindex(pd.MultiIndex.from_product([SETUPS, RUNS])).to_string())
print("\n== (configuración - full_system) por cubeta, pp del capital ==\n", DEC.to_string(index=False))
print("\n== Media de 3 ejecuciones de la descomposición ==\n", DEC.groupby("setup")[BUCKETS + ["total"]].mean().reindex(SETUPS[1:]).to_string())
sep = C[C.bucket == "Sep (12/16)"][["setup", "run", "entry_signal", "entry_exec", "entry_price", "shares_bought", "pnl"]].copy()
sep["pnl_por_accion"] = sep.pnl / sep.shares_bought
print("\n== Ciclos de septiembre (entrada, precio, acciones, P&L por acción) ==\n", sep.to_string(index=False))
