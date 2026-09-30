# Ficha de preparativos — Fase 4 [EN CURSO: 4.1 sin lanzar]
*(borrador para Capítulo 4 — Marco Metodológico, y para anexo técnico; punto de partida para escribir `run_all.py` en una conversación nueva)*

## 0. Estado en una línea

**El experimento real (20 backtests, periodo completo) NO se ha lanzado.** Lo ejecutado hasta ahora son pruebas de 6 días (2024-08-01 → 2024-08-08) sobre NVDA con `full_system`, cuyo objetivo era validar la infraestructura de ejecución. La infraestructura está validada; falta `run_all.py`, la limpieza previa y el lanzamiento.

## 1. Qué se ha hecho en estos preparativos

Respecto a lo previsto en el cronograma para 4.1, se añadió infraestructura que no estaba planificada, surgida de discutir cómo tratar los fallos y los informes de analista:

1. **Recuperación ante fallos con replay** (ver §3).
2. **Registro persistente de outputs** de analistas y trader por día (`*.days.jsonl`).
3. **Log de llamadas con tokens y coste reales** (`calls.jsonl`).
4. **Corrección de un bug de coste** del monitor de FINSABER con la Responses API (ver §4).

Ficheros modificados (versiones finales, a sustituir en el repo `C:\Users\pablo\TFM`):

| Fichero | Cambio |
|---|---|
| `strategy/ablation_strategy.py` | `DayFailure`, replay desde log, log de días, hook de fallo forzado, contexto de llamadas |
| `run_backtest.py` | `run_with_recovery()`, `ManualReviewRequired`, flag `--resume`, `--run-log-dir` |
| `llm/call_llm.py` | `configure_call_log()`, `set_call_context()`, `_record_cost()` (coste con `input_tokens`) |

`strategy/orchestrator.py` no se modificó. Su `OrchestratorResult` expone `analyst_reports` (lista de dicts), `trader_output`, `status` y `failed_role`.

## 2. Decisiones de diseño tomadas (con justificación)

| # | Decisión | Justificación |
|---|---|---|
| D1 | **No se cachean ni reutilizan informes de analista entre configuraciones.** Cada configuración llama a sus analistas de forma independiente. | Reutilizar el informe de un run en otro sería quedarse con una muestra arbitraria del espacio de outputs posibles de un LLM con `temperature=0.2`; no elimina el no determinismo, solo lo oculta. Se documenta como limitación (una sola ejecución/semilla, ya prevista para el Capítulo 6). |
| D2 | **Se registran todos los outputs** (informes de analista + decisión del trader) por (config, ticker, fecha), no solo uno. | Coste despreciable; sirve para el QA de 4.2 y como material del análisis cualitativo de 5.2. |
| D3 | **Fallo de un día ⇒ pausa, log y reanudación desde ese día; el resultado final tiene 0 días fallidos.** Sin umbral de re-ejecución. | Un día fallido no genera orden (equivale a un `hold`) y contaminaría la comparación entre configuraciones. Reintentar el día completo mantiene la comparabilidad. |
| D4 | **Tope: 3 fallos consecutivos en el mismo día ⇒ `ManualReviewRequired`** y se detiene. | Evita bucles infinitos y gasto descontrolado ante fallos sistemáticos (cuota agotada, caída del endpoint) y avisa de que hace falta revisión manual. |
| D5 | **Informes parciales del día fallido: se descartan** y se rehace el día completo. Se conservan solo en `failures.jsonl`. | Coherente con la decisión de 3.4: el trader nunca se llama con un subconjunto de informes. |
| D6 | **Mecanismo de reanudación: relanzar desde el principio y reproducir (replay) los días ya completados desde el log**, sin llamar al LLM. | Guardar/restaurar el estado interno de `FINSABERFrameworkHelper` es frágil y no se sabe si su `run()` admite arrancar a mitad. Con replay, FINSABER reconstruye cartera y órdenes por sí mismo; el resultado es idéntico al de un backtest sin fallo (verificado, §5). |
| D7 | **Seguridad del replay:** antes de reproducir un día se compara el estado de posición guardado con el actual; si difiere, se aborta con `RuntimeError`. | Detecta divergencias silenciosas que corromperían la comparación. |
| D8 | **El coste se suma desde `calls.jsonl`**, no desde el monitor en memoria ni desde `metrics.json`. | El contador en memoria se pierde al reiniciar un backtest; el log persiste e incluye los intentos fallidos (§4). |
| D9 | **Sin `--resume`, un backtest empieza limpio** (borra el `days.jsonl` de ese par config/ticker). Con `--resume` conserva el log. | Evita que un backtest nuevo reproduzca outputs de una ejecución anterior. |

## 3. Mecanismo de recuperación (referencia para `run_all.py`)

**Hallazgo verificado (salida del análisis del código de `finsaber==2.0.1`):** FINSABER **no captura excepciones** de `on_data()`. La llamada `strategy.on_data(...)` (`backtest_framework_iso.py`, línea 244) no está rodeada de `try/except`; los `except` de `finsaber.py` solo cubren `InsufficientTrainingDataException` y `(AttributeError, IndexError)` en otros puntos. Una excepción lanzada en `on_data()` sale de `framework.run()` y de `run_iterative_tickers()`.

Flujo:

1. `AblationStrategy.on_data()` detecta `status == "failed"` del orquestador → escribe el fallo en `failures.jsonl` → lanza `DayFailure`. La instancia queda «abortada»: cualquier `on_data()` posterior relanza el error sin llamar al LLM.
2. `run_with_recovery()` captura cualquier excepción y decide si hubo fallo **consultando `failures.jsonl` (línea con el `attempt` actual)**, no por el tipo de excepción (robusto si FINSABER envolviera el error en otra versión).
3. Borra los artefactos parciales (`<log_base_dir>/<setup>/*/*/<ticker>/`) y relanza con `attempt+1`.
4. En el nuevo intento, los días presentes en `days.jsonl` se reproducen; desde el día fallido se llama al LLM.
5. Contador de fallos consecutivos **por día**; al llegar a 3 → `ManualReviewRequired` (tras borrar artefactos parciales).
6. Con `resume=True` (o `--resume`), la numeración de intentos continúa desde el último intento registrado, no desde 1.

**API pública de `run_backtest.py`:**

```python
run_with_recovery(setup_name, ticker, eval_start, eval_end, preload_days=50,
                  run_log_dir=None, resume=False, max_day_failures=3) -> dict
# devuelve {periodo: {ticker: metrics}}  (con save_results=True)
class ManualReviewRequired(RuntimeError)
```

CLI: `python run_backtest.py --setup S --ticker T --eval-start YYYY-MM-DD --eval-end YYYY-MM-DD [--preload-days 50] [--resume] [--run-log-dir DIR]`.

## 4. Hallazgos técnicos nuevos (para el Capítulo 4 / anexo)

Se suman a los de las Fases 0 y 3 sobre `finsaber==2.0.1`:

1. **`add_openai_cost_from_response` es incompatible con la Responses API.** Lee `usage.prompt_tokens` (nombre de la Chat Completions API), que aquí no existe (es `input_tokens`); solo la salida tiene fallback a `output_tokens`. Resultado: el coste de entrada se contaba como 0 y el coste total quedaba infraestimado (~4,4×). **Solución:** `call_llm._record_cost()` llama a `add_llm_cost(model=AZURE_MODEL, prompt_tokens=input_tokens, completion_tokens=output_tokens, provider="openai")`, con el modelo fijado a la clave exacta de `OPENAI_PRICE`.
2. **Consecuencia para fases anteriores:** las cifras de coste de la Fase 3.6 estaban infraestimadas (24 llamadas de `full_system`: ≈ $0,0035 reportado frente a ≈ $0,015 real; estimación del experimento completo «$1-2» → ≈ $4). **Corregir en `entregable_fase3.md`.**
3. **Los roles de noticias pesan ~4× más que el resto:** sentimiento y fundamental ≈ 1.900 tokens de entrada por llamada (≈ $0,0009); técnico y trader ≈ 550 (≈ $0,00035).
4. **Discrepancia residual sin investigar:** en el intento 4 de la prueba de `--resume`, el monitor reportó $0,010124 y la suma de `calls.jsonl` $0,010250 (≈ 1,2 %). Irrelevante para el análisis (la fuente de coste es el log), pendiente de aclarar si se dispone de tiempo.

## 5. Validación realizada (NVDA, `full_system`, 2024-08-01 → 2024-08-08, Azure real)

| Prueba | Resultado |
|---|---|
| Ejecución normal tras los cambios | Sin regresión: −5,361 %, 2 trades, 0 `rejected_orders`, 24 llamadas |
| Fallo forzado el 08-05 (1 intento) | Intento 2 reproduce 08-01 y 08-02 sin llamadas HTTP; resultado idéntico al de la ejecución sin fallo; 24 llamadas en total (8 + 16) |
| Coste tras el parche | `total_llm_cost` ≈ $0,0152 (antes $0,0035) |
| Tope: fallo forzado 3 intentos | `ManualReviewRequired`; `failures.jsonl` = 3 líneas, `days.jsonl` = 2, `calls.jsonl` = 8 (los intentos 2 y 3 no llaman al LLM); sin `metrics.json` parcial |
| `--resume` tras el tope | Intento 4, replay de 2 días, resultado idéntico (−5,361 %, 2 trades, 0 rechazos); `calls.jsonl` = 24 líneas, $0,0153 |

**Nota:** la variabilidad entre ejecuciones (p. ej. `size` 0,40 vs 0,50 en pruebas de la Fase 3.6) es no determinismo del LLM, no un fallo del mecanismo de replay.

## 6. Formato de los logs (`<FINSABER_RESULTS_PATH>\_run_logs\`)

| Fichero | Una línea por | Campos |
|---|---|---|
| `<setup>__<ticker>.days.jsonl` | día evaluado completado | `date`, `status` (`"ok"`), `attempt`, `position_state`, `analyst_reports` (lista de dicts), `trader_output` |
| `failures.jsonl` | fallo de día | `setup`, `ticker`, `date`, `attempt`, `failed_role`, `partial_reports` |
| `calls.jsonl` | respuesta recibida de Azure (incluidas las que fallan al parsear) | `setup`, `ticker`, `date`, `attempt`, `system_head` (60 primeros caracteres del prompt de sistema; identifica el rol), `prompt_tokens`, `completion_tokens`, `cost` |

Advertencias:
- `failures.jsonl` y `calls.jsonl` **se acumulan entre ejecuciones** (solo el `days.jsonl` de un par se limpia al empezar sin `--resume`). Hay que vaciar `_run_logs` antes del lanzamiento real.
- Los días sin dato de precio (festivos) no se registran; el nº de días evaluados de un backtest = nº de líneas de su `days.jsonl`.
- Durante el preload no se llama al LLM, por lo que no aparecen en `calls.jsonl`.

## 7. Decisiones cerradas para `run_all.py` (4.1)

| # | Decisión | Justificación |
|---|---|---|
| R1 | **Periodo: `--eval-start 2024-08-01`, `--eval-end 2025-01-31`, `--preload-days 50`** (ficha de la Fase 1). No se activa la cláusula de flexibilidad de 2-3 meses. | El coste estimado es ≈ $4 y la recuperación automática elimina el trabajo manual que motivaba la cláusula. La cláusula queda como plan B si el tiempo de ejecución medido resulta inviable. |
| R2 | **Orden: por ticker, y dentro de cada ticker, `baseline_price_only` → `full_system` → `no_technical` → `no_sentiment` → `no_fundamental`.** Tickers en orden creciente de densidad de noticias 2024: **JPM (1.516) → JNJ (1.708) → XOM (2.210) → NVDA (20.211).** | Deja un ticker completo (las 5 configuraciones comparables) pronto, para revisar coste, ritmo y fallos antes de gastar el resto. `baseline` primero valida el trader solo (el más barato); `full_system` después valida el flujo completo. Empezar por el ticker más ligero minimiza el coste de un problema imprevisto. *(Corrige mi afirmación anterior de que JNJ era el de menor densidad: lo es JPM.)* |
| R3 | **Ejecución estrictamente secuencial.** | ≈ 7.000 llamadas en total; con 100 RPM de cuota la paralelización no compensa y complica el log y el análisis de fallos. |
| R4 | **Marcador propio de completado por par** (fichero en `_run_logs\done\<setup>__<ticker>.json`, con resumen), escrito solo tras terminar con éxito. `run_all.py` salta los pares con marcador. | `metrics.json` no es un indicador fiable (puede haber restos de pruebas); el marcador propio es la fuente de verdad. Se puede cruzar con la existencia de `metrics.json` como comprobación. |
| R5 | **`run_all.py` llama siempre a `run_with_recovery(..., resume=True)`.** | Si el proceso se interrumpe (Ctrl+C, corte de luz) a mitad de un par, al relanzar reutiliza su `days.jsonl` en vez de repetir llamadas. Un par nuevo no tiene log, así que no afecta. Los pares completados no se relanzan gracias a R4. |
| R6 | **Ante `ManualReviewRequired` se detiene todo el lanzamiento** (código de salida ≠ 0), sin pasar al siguiente par. | Un día que falla 3 veces seguidas indica un problema sistemático (cuota, endpoint, prompt); continuar gastaría tokens sobre una causa no resuelta. |
| R7 | **Guardia de seguridad al arrancar:** abortar si `TFM_FORCE_FAIL_DATE` o `TFM_FORCE_FAIL_ATTEMPTS` están definidas. | Son interruptores de prueba; si quedaran en el entorno forzarían fallos silenciosamente en el experimento real. No van en `.env`. |
| R8 | **Tope de gasto `--max-cost` (por defecto $10),** comprobado tras cada backtest sumando `calls.jsonl`. | ≈ 2,5× la estimación (≈ $4); protege ante un gasto descontrolado imprevisto. |
| R9 | **Resumen por par en `_run_logs\run_all_summary.csv`**: setup, ticker, intentos, días evaluados, días con fallo, trades, `rejected_orders`, retorno total, coste (suma de `calls.jsonl` filtrada por par), tiempo de reloj. | Da el criterio de QA de 4.2 sin abrir cada fichero y permite vigilar los primeros backtests. |
| R10 | **Opciones de CLI:** `--tickers`, `--setups`, `--eval-start`, `--eval-end`, `--preload-days`, `--max-cost`, `--dry-run` (imprime el plan y el estado de cada par sin ejecutar). | Permite lanzar por partes y revisar el plan antes de gastar. |

**Estimación de volumen y coste (a confirmar con los primeros backtests):**

| Config | Llamadas/día | Coste/día (≈) |
|---|---|---|
| `full_system` | 4 | $0,0025 |
| `no_technical` | 3 | $0,0022 |
| `no_sentiment` | 3 | $0,0016 |
| `no_fundamental` | 3 | $0,0016 |
| `baseline_price_only` | 1 | $0,0002 (sin medir con este parche) |
| **Total por ticker-día** | **14** | **≈ $0,008** |

≈ 125 días × 4 tickers ⇒ ≈ 7.000 llamadas y **≈ $4** en total. El **tiempo de reloj no está medido**: registrarlo en los primeros backtests (R9) y extrapolar antes de dejar correr el resto. Con llamadas secuenciales, el ritmo previsible queda muy por debajo de los 100 RPM.

## 8. Checklist previo al lanzamiento (a hacer antes de `run_all.py`) (Todos Hechos)

- [X] Sustituir en el repo `ablation_strategy.py`, `run_backtest.py` y `call_llm.py` por las versiones finales.
- [X] Confirmar que `TFM_FORCE_FAIL_DATE` y `TFM_FORCE_FAIL_ATTEMPTS` no están definidas (PowerShell: `Remove-Item Env:TFM_FORCE_FAIL_DATE`, `Remove-Item Env:TFM_FORCE_FAIL_ATTEMPTS`).
- [X] Borrar `C:\finsaber-data\results\_run_logs`.
- [X] Vaciar o mover a respaldo los resultados de pruebas bajo `C:\finsaber-data\results\` (`full_system`, `baseline_price_only`, …).
- [X] Commit en Git con los tres ficheros modificados.

## 9. Notas para fases posteriores

**4.2 (control de calidad).** Los criterios ya no requieren umbral de fallos: la política D3 garantiza 0 días fallidos. El QA consiste en revisar `failures.jsonl` (cuántos reintentos hubo y por qué rol), `rejected_orders` y el nº de días evaluados por backtest (≈ 125).

**4.3 (consolidación).**
- **Coste:** usar la suma de `calls.jsonl` por par (config, ticker); **no** `total_llm_cost` de `metrics.json` (en un backtest recuperado solo recoge el último intento).
- `metrics.json` mezcla unidades: `total_return`/`annual_return`/`annual_volatility` en fracción; `max_drawdown` en porcentaje positivo. No incluye Calmar ni nº de operaciones (Calmar se calcula a mano con unidades consistentes; las operaciones se cuentan desde `trades.csv`).
- Delta frente a `full_system` y agregación por configuración (promedio de los 4 tickers + desglose por ticker), según la Fase 1.4.

**Memoria (Capítulos 4 y 6).** Documentar: (a) el mecanismo de recuperación y su justificación (D3-D7), (b) los hallazgos técnicos de §4, (c) que la ejecución es de una sola semilla y que los outputs de los analistas no se comparten entre configuraciones (D1), y (d) el hecho de que los días reintentados generan llamadas extra cuyos outputs descartados quedan en `failures.jsonl`.

## 10. Pendientes abiertos

- **Escribir `run_all.py`** (siguiente paso, con las decisiones R1-R10).
- Corregir cifras de coste en `entregable_fase3.md` (§4, punto 2).

## 11. Material a aportar en la nueva conversación

`entregable_fase4_preparativos.md` (este fichero), `run_backtest.py` (versión final), `config/ablation.yaml` (nombres y roles de las 5 configuraciones) y el cronograma. Los entregables de las Fases 0-3 ya están en el proyecto.
