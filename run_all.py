"""run_all.py — Fase 4.1: lanza los 20 backtests (5 configuraciones x 4 tickers).

Decisiones R1-R10 de entregable_fase4_preparativos.md:
  R1 periodo 2024-08-01 -> 2025-01-31, preload 50 sesiones
  R2 orden: por ticker (JPM, JNJ, XOM, NVDA); dentro, baseline -> full -> no_technical -> no_sentiment -> no_fundamental
  R3 secuencial | R4 marcador de completado por par | R5 resume=True siempre
  R6 ManualReviewRequired detiene todo | R7 guardia de variables de fallo forzado
  R8 tope de gasto --max-cost | R9 resumen CSV por par | R10 CLI (--dry-run, etc.)

Ejecutar desde la raíz del repo (C:\\Users\\pablo\\TFM) con el venv activado:
  python run_all.py --dry-run
  python run_all.py --tickers JPM            # solo un ticker
  python run_all.py                          # experimento completo

Códigos de salida: 0 OK | 2 ManualReviewRequired | 3 tope de gasto | 4 error inesperado
"""
import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

DEFAULT_TICKERS = ["JPM", "JNJ", "XOM", "NVDA"]
DEFAULT_SETUPS = ["baseline_price_only", "full_system", "no_technical",
                  "no_sentiment", "no_fundamental"]
FORCE_FAIL_VARS = ("TFM_FORCE_FAIL_DATE", "TFM_FORCE_FAIL_ATTEMPTS")
ABLATION_YAML = Path(__file__).resolve().parent / "config" / "ablation.yaml"

SUMMARY_FIELDS = ["timestamp", "setup", "ticker", "attempts", "days_evaluated",
                  "day_failures", "trades", "rejected_orders", "total_return",
                  "cost_usd", "wall_seconds"]


def yaml_setup_names() -> list[str]:
    with ABLATION_YAML.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return [c["setup_name"] for c in cfg["configurations"]]


def read_jsonl(path: Path):
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8", newline="") as f:
        return max(sum(1 for _ in f) - 1, 0)  # menos cabecera


def pair_cost(calls, setup=None, ticker=None) -> float:
    return sum(c.get("cost", 0.0) for c in calls
               if (setup is None or c.get("setup") == setup)
               and (ticker is None or c.get("ticker") == ticker))


def find_artifact_dir(results_dir: Path, setup: str, ticker: str):
    # log_base_dir/setup/strategy/date_from_date_to/ticker/
    matches = sorted(results_dir.glob(f"{setup}/*/*/{ticker}"))
    return matches[-1] if matches else None


def extract_total_return(result, ticker):
    """result = {periodo: {ticker: metrics}}"""
    try:
        for period_metrics in result.values():
            if ticker in period_metrics:
                return period_metrics[ticker].get("total_return")
    except AttributeError:
        pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    ap.add_argument("--setups", nargs="+", default=DEFAULT_SETUPS)
    ap.add_argument("--eval-start", default="2024-08-01")
    ap.add_argument("--eval-end", default="2025-01-31")
    ap.add_argument("--preload-days", type=int, default=50)
    ap.add_argument("--max-cost", type=float, default=10.0)
    ap.add_argument("--show-figures", action="store_true",
                    help="Abrir también la equity curve en ventana NO bloqueante (siempre se guarda el PNG)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # R7: guardia de interruptores de prueba
    active = [v for v in FORCE_FAIL_VARS if os.environ.get(v)]
    if active:
        sys.exit(f"ABORTADO: variables de fallo forzado definidas: {active}. "
                 "Elimínalas antes del experimento real.")

    # Validar setups contra config/ablation.yaml
    valid = yaml_setup_names()
    unknown = [s for s in args.setups if s not in valid]
    if unknown:
        sys.exit(f"setup(s) desconocido(s): {unknown}. Válidos en ablation.yaml: {valid}")

    results_dir = Path(os.environ["FINSABER_RESULTS_PATH"])
    run_log_dir = results_dir / "_run_logs"
    done_dir = run_log_dir / "done"
    summary_path = run_log_dir / "run_all_summary.csv"

    pairs = [(s, t) for t in args.tickers for s in args.setups]  # R2

    def marker(s, t):
        return done_dir / f"{s}__{t}.json"

    pending = [(s, t) for s, t in pairs if not marker(s, t).exists()]
    print(f"Periodo: {args.eval_start} -> {args.eval_end} (preload {args.preload_days})")
    print(f"Logs: {run_log_dir}")
    print(f"Figuras: {results_dir / '_figures'} ({'+ ventana no bloqueante' if args.show_figures else 'solo PNG, sin ventanas'})")
    print(f"Pares totales: {len(pairs)} | completados: {len(pairs) - len(pending)} | pendientes: {len(pending)}")
    for s, t in pairs:
        print(f"  [{'HECHO' if marker(s, t).exists() else 'pend.'}] {t:5s} {s}")
    if args.dry_run:
        return

    # Import tardío: evita cargar FINSABER/Azure en --dry-run
    from run_backtest import ManualReviewRequired, _last_attempt, run_with_recovery
    from strategy.ablation_strategy import days_log_path, failures_log_path, jsonl_read

    run_log_dir.mkdir(parents=True, exist_ok=True)
    done_dir.mkdir(parents=True, exist_ok=True)
    calls_path = run_log_dir / "calls.jsonl"

    t_global = time.perf_counter()
    for i, (setup, ticker) in enumerate(pending, 1):
        print(f"\n=== [{i}/{len(pending)}] {setup} / {ticker} ===", flush=True)
        t0 = time.perf_counter()
        try:
            result = run_with_recovery(
                setup, ticker, args.eval_start, args.eval_end,
                preload_days=args.preload_days,
                run_log_dir=run_log_dir,
                resume=True,  # R5
                show_figures=args.show_figures,
            )
        except ManualReviewRequired as e:  # R6
            print(f"\nREVISIÓN MANUAL REQUERIDA en {setup}/{ticker}: {e}", file=sys.stderr)
            print("Detenido. Revisa failures.jsonl y relanza (retomará con resume).",
                  file=sys.stderr)
            sys.exit(2)
        except Exception as e:  # noqa: BLE001
            print(f"\nERROR INESPERADO en {setup}/{ticker}: {type(e).__name__}: {e}", file=sys.stderr)
            print("Detenido. El par no está marcado como completado; se puede relanzar.",
                  file=sys.stderr)
            sys.exit(4)
        wall = time.perf_counter() - t0

        # Resumen del par (R9)
        calls = read_jsonl(calls_path)
        fails = [f for f in jsonl_read(failures_log_path(run_log_dir))
                 if f.get("setup") == setup and f.get("ticker") == ticker]
        days = jsonl_read(days_log_path(run_log_dir, setup, ticker))
        art = find_artifact_dir(results_dir, setup, ticker)
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "setup": setup, "ticker": ticker,
            "attempts": _last_attempt(run_log_dir, setup, ticker),
            "days_evaluated": len(days),
            "day_failures": len(fails),
            "trades": count_csv_rows(art / "trades.csv") if art else "",
            "rejected_orders": count_csv_rows(art / "rejected_orders.csv") if art else "",
            "total_return": extract_total_return(result, ticker),
            "cost_usd": round(pair_cost(calls, setup, ticker), 6),
            "wall_seconds": round(wall, 1),
        }
        new_file = not summary_path.exists()
        with summary_path.open("a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
            if new_file:
                w.writeheader()
            w.writerow(row)

        # R4: marcador solo tras éxito
        marker(setup, ticker).write_text(json.dumps(row, indent=2), encoding="utf-8")
        print(f"OK  días={row['days_evaluated']} fallos={row['day_failures']} "
              f"trades={row['trades']} rej={row['rejected_orders']} "
              f"ret={row['total_return']} coste=${row['cost_usd']} t={row['wall_seconds']}s")

        # R8: tope de gasto acumulado
        total = pair_cost(calls)
        print(f"Coste acumulado: ${total:.4f} / ${args.max_cost}")
        if total > args.max_cost:
            print("TOPE DE GASTO SUPERADO. Detenido.", file=sys.stderr)
            sys.exit(3)

    print(f"\nLanzamiento completo en {(time.perf_counter() - t_global) / 60:.1f} min.")


if __name__ == "__main__":
    main()