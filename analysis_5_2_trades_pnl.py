"""Fase 5.2 - P&L por ciclo de posición para un ticker y vista de operaciones alrededor de un episodio.
Generaliza analysis_5_2_nvda_pnl.py (que se conserva tal cual para reproducir el resultado de NVDA).
Uso:
  python analysis_5_2_trades_pnl.py --ticker XOM --trades-dir analysis\\trades_xom --consolidated <rep1.csv> <rep2.csv> <rep3.csv> --episode-date 2024-09-26 --out analysis\\5_2_xom
Ciclo = periodo entre dos momentos con 0 acciones. Cubeta = mes de la señal de entrada del ciclo (sin fechas escritas a mano).
P&L del ciclo = ingresos de ventas - coste de compras - comisiones, con precios y comisiones de trades.csv.
Validación: P&L total frente al retorno consolidado (capital inicial 100.000 USD).
"""
import argparse, glob, os, re
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--ticker", required=True)
ap.add_argument("--trades-dir", required=True)
ap.add_argument("--consolidated", nargs=3, required=True)
ap.add_argument("--episode-date", default=None, help="fecha de señal del episodio (YYYY-MM-DD)")
ap.add_argument("--episode-ops", type=int, default=6, help="nº de operaciones a mostrar desde la fecha del episodio")
ap.add_argument("--out", required=True)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
CAP = 100000.0; TK = a.ticker
RUNS = ["rep1", "rep2", "rep3"]; SETUPS = ["full_system", "no_technical", "no_sentiment", "no_fundamental"]

cycles, ops = [], []
files = sorted(glob.glob(os.path.join(a.trades_dir, "**", f"trades_*__{TK}__rep*.csv"), recursive=True))
if len(files) != 12: raise SystemExit(f"se esperaban 12 ficheros y hay {len(files)} en {a.trades_dir}")
for f in files:
    m = re.match(rf"trades_(.+)__{TK}__(rep\d)\.csv", os.path.basename(f)); s, r = m.group(1), m.group(2)
    t = pd.read_csv(f).sort_values(["execution_date", "signal_date"]).reset_index(drop=True)
    pos = 0; cur = None
    for _, x in t.iterrows():
        q = int(x.quantity); sg = 1 if x.type == "buy" else -1
        pos_before = pos
        if pos == 0:
            cur = dict(setup=s, run=r, entry_signal=str(x.signal_date)[:10], entry_exec=str(x.execution_date)[:10],
                       entry_price=x.price, shares_bought=0, n_sells=0, pnl=0.0)
        cur["pnl"] += -sg * q * x.price - x.commission
        if x.type == "buy": cur["shares_bought"] += q
        else: cur["n_sells"] += 1
        pos += sg * q
        ops.append(dict(setup=s, run=r, signal_date=str(x.signal_date)[:10], execution_date=str(x.execution_date)[:10], type=x.type,
                        price=x.price, quantity=q, shares_before=pos_before, shares_after=pos, commission=x.commission))
        if pos == 0:
            cur["exit_exec"] = str(x.execution_date)[:10]; cycles.append(cur); cur = None
    if pos != 0: raise SystemExit(f"{s}/{r}: posición abierta al final ({pos})")
C = pd.DataFrame(cycles); O = pd.DataFrame(ops)
C["bucket"] = C.entry_signal.str[:7]; C["pnl_pct_capital"] = C.pnl / CAP * 100
C.to_csv(f"{a.out}/{TK}_cycles.csv", index=False); O.to_csv(f"{a.out}/{TK}_operations.csv", index=False)

cons = pd.concat([pd.read_csv(p).assign(run=r) for r, p in zip(RUNS, a.consolidated)])
cons = cons[(cons.ticker == TK) & cons.setup.isin(SETUPS)].set_index(["setup", "run"])
idx = pd.MultiIndex.from_product([SETUPS, RUNS], names=["setup", "run"])
tot = C.groupby(["setup", "run"]).agg(pnl=("pnl", "sum"), ciclos=("pnl", "size")).reindex(idx)
tot["retorno_trades_pct"] = tot.pnl / CAP * 100
tot["retorno_consolidado_pct"] = cons.total_return.reindex(idx) * 100
tot["diferencia_pp"] = tot.retorno_trades_pct - tot.retorno_consolidado_pct
tot.to_csv(f"{a.out}/{TK}_pnl_validation.csv")

# descomposición (configuración - full_system) por mes de entrada
pv = C.pivot_table(index=["setup", "run"], columns="bucket", values="pnl", aggfunc="sum").fillna(0.0) / CAP * 100
dec = []
for s in SETUPS[1:]:
    for r in RUNS:
        d = pv.loc[(s, r)] - pv.loc[("full_system", r)]
        dec.append(dict(setup=s, run=r, **d.to_dict(), total=d.sum()))
DEC = pd.DataFrame(dec); DEC.to_csv(f"{a.out}/{TK}_decomposition_vs_full.csv", index=False)
pv.assign(total=pv.sum(axis=1)).to_csv(f"{a.out}/{TK}_pnl_by_entry_month.csv")

pd.set_option("display.width", 230, "display.max_columns", 40, "display.float_format", lambda v: f"{v:,.2f}")
print("== Validación P&L trades.csv vs consolidado ==\n", tot.to_string())
print("\n== P&L por mes de entrada (% capital) ==\n", pv.assign(total=pv.sum(axis=1)).reindex(idx).to_string())
print("\n== (configuración - full_system) por mes de entrada, media de 3 ejecuciones (pp) ==\n",
      DEC.groupby("setup").mean(numeric_only=True).reindex(SETUPS[1:]).to_string())
if a.episode_date:
    ed = a.episode_date
    print(f"\n== Operaciones desde la señal {ed} (primeras {a.episode_ops}), por configuración y ejecución ==")
    for s in SETUPS:
        for r in RUNS:
            x = O[(O.setup == s) & (O.run == r) & (O.signal_date >= ed)].head(a.episode_ops)
            held = O[(O.setup == s) & (O.run == r) & (O.signal_date < ed)]
            sh = int(held.shares_after.iloc[-1]) if len(held) else 0
            print(f"\n{s} / {r} (acciones en cartera antes de {ed}: {sh})")
            print(x[["signal_date", "execution_date", "type", "price", "quantity", "shares_after"]].to_string(index=False))

if a.episode_date:
    # efecto aproximado de la operación del episodio: acciones vendidas x (precio de la siguiente venta de la misma ejecución - precio de venta)
    # (positivo = esperar hasta la siguiente venta habría dado más). Es una aproximación con los precios de trades.csv, no un contrafactual.
    rows = []
    for s_ in SETUPS:
        for r in RUNS:
            x = O[(O.setup == s_) & (O.run == r)].reset_index(drop=True)
            on = x[(x.signal_date == ed)]
            nxt = x[x.signal_date > ed]
            nxt_sell = nxt[nxt.type == "sell"].head(1)
            row = dict(setup=s_, run=r, op_en_episodio=(on.type.iloc[0] if len(on) else "hold"),
                       acciones_op=int(on.quantity.iloc[0]) if len(on) else 0,
                       precio_op=float(on.price.iloc[0]) if len(on) else None,
                       siguiente_venta_signal=(nxt_sell.signal_date.iloc[0] if len(nxt_sell) else None),
                       siguiente_venta_precio=(float(nxt_sell.price.iloc[0]) if len(nxt_sell) else None))
            if len(on) and on.type.iloc[0] == "sell" and len(nxt_sell):
                row["coste_aprox_de_vender_antes_usd"] = row["acciones_op"] * (row["siguiente_venta_precio"] - row["precio_op"])
                row["coste_aprox_pct_capital"] = row["coste_aprox_de_vender_antes_usd"] / CAP * 100
            rows.append(row)
    EP = pd.DataFrame(rows); EP.to_csv(f"{a.out}/{TK}_episode_effect.csv", index=False)
    print("\n== Efecto aproximado de la operación del episodio ==\n", EP.to_string(index=False))

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    cols = {"full_system": "#1f77b4", "no_technical": "#d62728", "no_sentiment": "#2ca02c", "no_fundamental": "#ff7f0e"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), sharey=True)
    for ax, r in zip(axes, RUNS):
        for s_ in SETUPS:
            x = O[(O.setup == s_) & (O.run == r)].copy(); x["d"] = pd.to_datetime(x.execution_date)
            for typ, mk in (("sell", "v"), ("buy", "^")):
                y = x[x.type == typ]
                ax.scatter(y.d, y.price, marker=mk, s=18 + 90 * (y.quantity / O.quantity.max()), color=cols[s_], alpha=.8,
                           label=s_ if typ == "sell" else None)
        ax.axvline(pd.Timestamp(ed), color="k", ls="--", lw=1); ax.set_title(r); ax.grid(alpha=.3)
        ax.tick_params(axis="x", rotation=30)
    axes[0].set_ylabel(f"Precio de ejecución ({TK})"); axes[0].legend(fontsize=7)
    fig.suptitle(f"{TK}: operaciones (▼ venta, ▲ compra; tamaño = acciones). Línea discontinua: señal del episodio {ed}")
    fig.tight_layout(); fig.savefig(f"{a.out}/{TK}_episode_operations.png", dpi=160)
