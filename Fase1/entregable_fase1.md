# Ficha de ejecución y consolidación — Fase 4 [CERRADA]
*(borrador para Capítulo 4 — Marco Metodológico, para el anexo técnico y como base del Capítulo 5; sustituye a `entregable_fase4_preparativos.md`, que queda como histórico de decisiones)*

## 0. Estado en una línea

**Los 20 backtests (5 configuraciones × 4 tickers, 2024-08-01 → 2025-01-31) se ejecutaron completos: 126 sesiones evaluadas cada uno, 0 días fallidos, 0 reintentos, 0 órdenes rechazadas; coste LLM total $3,54.** Los resultados están consolidados con una referencia externa buy-and-hold en `Fase4/consolidated/`. Lista para la Fase 5.

---

## 4.1 Ejecución de las 5 configuraciones [CERRADO]

### Parámetros del experimento

| Parámetro | Valor |
|---|---|
| Periodo de evaluación | 2024-08-01 → 2025-01-31 (126 sesiones por backtest) |
| Preload de histórico | 50 sesiones antes del inicio (carga desde 2024-05-13), sin operar; solo para que el analista técnico tenga indicadores completos el primer día |
| Tickers | JPM, JNJ, XOM, NVDA |
| Configuraciones | `baseline_price_only`, `full_system`, `no_technical`, `no_sentiment`, `no_fundamental` |
| Capital inicial | 100.000 USD por backtest (sin pool compartido entre tickers) |
| Ejecución de órdenes | `execution_timing="next_open"` (señal al cierre del día, fill a la apertura siguiente) |
| Costes del motor | comisión 0,0049 USD/acción (mín. 0,99), slippage 0, tope de liquidez desactivado |
| Modelo | `gpt-4.1-mini` (Azure AI Foundry, Responses API), `temperature=0.2`, una sola ejecución (sin semillas múltiples) |
| Lanzador | `run_all.py` (secuencial), que invoca `run_backtest.run_with_recovery(..., resume=True)` |

**Calendario real:** JPM el 30-sep-2026; JNJ, XOM y NVDA el 1-oct-2026. Orden R2 (de menor a mayor densidad de noticias de 2024): JPM → JNJ → XOM → NVDA; dentro de cada ticker, baseline → full → no_technical → no_sentiment → no_fundamental.

### Infraestructura añadida respecto al cronograma

Surgida al discutir cómo tratar los fallos y los informes de los analistas; no estaba planificada para 4.1:

1. Recuperación ante fallos con replay desde log.
2. Registro persistente de los outputs de analistas y trader por día (`<setup>__<ticker>.days.jsonl`).
3. Log de llamadas con tokens y coste real (`calls.jsonl`).
4. Corrección del cálculo de coste con la Responses API (ver hallazgo 1).
5. Lanzador `run_all.py` con marcadores de completado, tope de gasto y resumen por par.
6. Parche de la figura de equity para que no bloquee la ejecución (ver hallazgo 2).

### Decisiones de diseño de la recuperación (D1-D9)

| # | Decisión | Justificación |
|---|---|---|
| D1 | No se cachean ni reutilizan informes de analista entre configuraciones | Reutilizar un informe sería elegir una muestra arbitraria del espacio de outputs de un LLM no determinista; se documenta como limitación (una sola ejecución) |
| D2 | Se registran todos los outputs (analistas y trader) por (config, ticker, fecha) | Coste despreciable; sirve para el QA y para el análisis cualitativo (5.2) |
| D3 | Fallo de un día ⇒ pausa, log y reanudación desde ese día; el resultado final tiene 0 días fallidos | Un día fallido equivale a un `hold` y contaminaría la comparación entre configuraciones |
| D4 | 3 fallos consecutivos el mismo día ⇒ `ManualReviewRequired` | Evita bucles y gasto descontrolado ante fallos sistemáticos |
| D5 | Los informes parciales del día fallido se descartan y se rehace el día completo (solo quedan en `failures.jsonl`) | El trader nunca se llama con un subconjunto de los informes de su configuración |
| D6 | Reanudación por replay: se relanza desde el principio reproduciendo desde el log los días ya completados, sin llamar al LLM | Guardar y restaurar el estado interno del framework es frágil; con replay FINSABER reconstruye cartera y órdenes por sí mismo |
| D7 | Antes de reproducir un día se compara el estado de posición guardado con el actual; si difiere, `RuntimeError` | Detecta divergencias silenciosas |
| D8 | El coste se suma desde `calls.jsonl`, no desde el monitor en memoria ni desde `metrics.json` | El contador en memoria se pierde al reiniciar y el log incluye los intentos fallidos |
| D9 | Sin `--resume`, un backtest empieza limpio; con `--resume` conserva el log | Evita reproducir outputs de una ejecución anterior |

**Hallazgo verificado en el código de `finsaber==2.0.1`:** el framework no captura excepciones de `on_data()`; una excepción sale de `framework.run()`. Por eso `run_with_recovery()` decide si hubo fallo consultando `failures.jsonl` y no por el tipo de excepción.

**Resultado en producción:** el mecanismo se validó en pruebas de 6 días con fallos forzados (replay idéntico al backtest sin fallo, tope de 3 intentos, `--resume`), pero **no llegó a dispararse en el experimento real** (no existe `failures.jsonl`).

### Decisiones de lanzamiento (R1-R10)

| # | Decisión |
|---|---|
| R1 | Periodo completo (no se activó la cláusula de flexibilidad de 2-3 meses de la Fase 1.2) |
| R2 | Orden por ticker ascendente en densidad de noticias, para validar coste, ritmo y fallos con un ticker completo antes de gastar el resto |
| R3 | Ejecución estrictamente secuencial (≈ 7.000 llamadas; la paralelización no compensa con 100 RPM y complica el log) |
| R4 | Marcador de completado por par en `_run_logs\done\`, escrito solo tras terminar con éxito; `run_all.py` salta los pares con marcador |
| R5 | `resume=True` siempre, para poder interrumpir y relanzar sin repetir llamadas |
| R6 | `ManualReviewRequired` detiene todo el lanzamiento (código de salida 2) |
| R7 | Guardia al arrancar: se aborta si `TFM_FORCE_FAIL_DATE` o `TFM_FORCE_FAIL_ATTEMPTS` están definidas |
| R8 | Tope de gasto `--max-cost` (por defecto $10), sobre el acumulado de `calls.jsonl`; código de salida 3 |
| R9 | Resumen por par en `run_all_summary.csv` (intentos, días, fallos, trades, rechazos, retorno, coste, tiempo) |
| R10 | CLI: `--tickers`, `--setups`, `--eval-start`, `--eval-end`, `--preload-days`, `--max-cost`, `--dry-run`, `--show-figures` |

Los setups de `--setups` se validan contra `config/ablation.yaml`; cualquier excepción distinta de `ManualReviewRequired` termina con código 4 sin marcar el par.

### Hallazgos técnicos nuevos (para el Capítulo 4 / anexo)

Se suman a los de las Fases 0 y 3 sobre `finsaber==2.0.1`:

1. **`add_openai_cost_from_response` es incompatible con la Responses API.** Lee `usage.prompt_tokens` (Chat Completions), que no existe (aquí es `input_tokens`), de modo que el coste de entrada contaba como 0 y el total quedaba infraestimado ≈ 4,4×. Solución: `call_llm._record_cost()` llama a `add_llm_cost(...)` con `input_tokens` y `output_tokens`. Corregidas ya las cifras de la Fase 3.
2. **La figura de equity es bloqueante.** `run_iterative_tickers()` llama a `_plot_equity_curve()` cuando `silence=False`, y ese método termina en `plt.show()`: la ejecución se detenía hasta cerrar la ventana, y la figura no se guardaba. Solución: `_patch_equity_plot()` en `run_backtest.py` sustituye el método; guarda siempre el PNG en `<results>\_figures\<setup>__<ticker>.png` y no abre ventana por defecto (opción `--show-figures` para una ventana no bloqueante). Un fallo al dibujar no detiene el backtest. JPM (primer ticker) se ejecutó antes del parche y no tiene PNG de FINSABER; sus curvas salen igualmente de `equity_curve.csv` en la consolidación. Efecto lateral: los `wall_seconds` de JPM incluyen la espera por las ventanas.
3. **Las métricas de FINSABER se calculan sobre la ventana cargada, que incluye el preload.** Los `equity_curve.csv` empiezan el 2024-05-13 con ≈ 55 sesiones planas a 100.000 antes del 2024-08-01. Annual return, volatilidad, Sharpe y Sortino quedan diluidos; `total_return` y `max_drawdown` no cambian. Se recalculan en 4.3 sobre la ventana de evaluación.
4. **El coste LLM no se descuenta del equity.** En el baseline el retorno es exactamente 0 con $0,03 de gasto: `total_trading_cost` suma comisión + LLM, pero `final_value` no refleja este último. Los retornos son brutos de coste LLM, que se reporta aparte (≈ 0,0003-0,0003 % del capital por par; es despreciable).
5. **Los roles de noticias pesan ≈ 3× más que el resto** (media de tokens de entrada: macro 1.499, sentimiento 1.432, técnico 543, trader 467), con el presupuesto de 6.000 caracteres como tope. NVDA llega a ≈ 1.900 tokens en los roles de noticias; JPM, el ticker con menos noticias, a ≈ 1.100.
6. **Discrepancia residual sin investigar** (≈ 1,2 % entre el monitor y `calls.jsonl` en una prueba de `--resume`): irrelevante, porque el coste sale del log.

---

## 4.2 Control de calidad [CERRADO]

| Comprobación | Resultado |
|---|---|
| Pares completados | 20/20 (20 marcadores en `done\`, 20 filas en `run_all_summary.csv`) |
| Sesiones evaluadas | 126 en cada par |
| Fallos de día / reintentos (`attempts`) | 0 / 1 (todos en el primer intento); no existe `failures.jsonl` |
| `rejected_orders` | 0 en todos los pares |
| Llamadas LLM | 7.056 = 126 × 14 × 4, todas con `attempt=1`; por par: 126 (baseline), 504 (full), 378 (cada ablación) |
| Operaciones | 183 en las 20 configuraciones (+ 8 de los 4 buy-and-hold); el baseline no opera en ningún ticker |
| Coste LLM | $3,5434 (suma de `calls.jsonl` = suma del CSV), frente a ≈ $4 estimados y al tope de $10 |
| Tiempo de reloj | ≈ 3,8 h en total; por ticker: JPM 63,1 min (inflado por las ventanas), JNJ 57,0, XOM 56,0, NVDA 52,8 |
| Retornos exactamente 0 fuera del baseline | ninguno |

**Operaciones por par** (filas de `trades.csv`, incluida la liquidación final del motor si hay posición abierta):

| Ticker | baseline | full | no_technical | no_sentiment | no_fundamental |
|---|---|---|---|---|---|
| JPM | 0 | 5 | 4 | 5 | 7 |
| JNJ | 0 | 19 | 15 | 14 | 17 |
| XOM | 0 | 16 | 14 | 13 | 17 |
| NVDA | 0 | 11 | 10 | 7 | 9 |

La actividad depende del ticker (JPM opera poco, JNJ y XOM mucho) más que de la configuración, así que no es una anomalía de comparabilidad.

**Coste LLM por par (USD):**

| Ticker | baseline | full | no_technical | no_sentiment | no_fundamental | Total |
|---|---|---|---|---|---|---|
| JPM | 0,029 | 0,240 | 0,190 | 0,162 | 0,158 | 0,779 |
| JNJ | 0,029 | 0,270 | 0,222 | 0,177 | 0,174 | 0,872 |
| XOM | 0,029 | 0,274 | 0,226 | 0,179 | 0,176 | 0,884 |
| NVDA | 0,029 | 0,316 | 0,267 | 0,200 | 0,197 | 1,009 |
| **Total** | 0,117 | 1,100 | 0,905 | 0,718 | 0,704 | **3,543** |

**Coste por rol** (de `calls.jsonl`): macro/fundamental $1,144; sentimiento $1,103; trader $0,747; técnico $0,549.

---

## 4.3 Consolidación de resultados [CERRADO]

### Método (`consolidate_results.py`)

- **Métricas recalculadas sobre la ventana de evaluación** (≥ 2024-08-01) a partir de `equity_curve.csv`, con las fórmulas de FINSABER (`backtest_framework_iso.evaluate`) y rf = 3 %. Control: el `total_return` recalculado coincide con el de FINSABER con una diferencia máxima de 4,7 × 10⁻⁵ (la liquidación final del motor no queda reflejada en el último punto de la curva).
- **Calmar** = retorno anualizado / max drawdown (en fracción); queda `NaN` cuando el drawdown es 0 (baseline).
- **Coste LLM:** suma de `calls.jsonl` por par. **Nº de operaciones:** filas de `trades.csv`.
- **Agregación (Fase 1.4):** promedio de los 4 tickers por configuración, más desglose por ticker.
- **Deltas frente a `full_system`:** diferencia absoluta (configuración − `full_system`) en todas las métricas, en puntos porcentuales para retorno total, retorno anualizado y max drawdown, y en unidades propias para Sharpe, Calmar, nº de operaciones y coste (USD). Se descarta la variación relativa porque los retornos son cercanos a cero y a menudo negativos. **Contribución marginal de un rol = −delta de la configuración que lo elimina** (delta > 0 ⇒ el sistema mejora sin ese rol). *(Sustituye a la definición «porcentual» de la Fase 1.4.)*
- **Buy-and-hold (`run_buyhold.py`):** referencia externa ejecutada en el mismo motor y con las mismas convenciones (señal 2024-08-01, fill a la apertura del 2024-08-02, liquidación final el 2025-01-31, sin llamadas a LLM). Queda fuera de las medias y de los deltas; aparece en `consolidated_results.csv`, en la matriz de retornos y en las figuras. Cada buy-and-hold registra 2 operaciones (compra y liquidación final).

### Resultados consolidados (ventana de evaluación; métricas en %, salvo ratios)

**Retorno total por ticker y configuración (%):**

| Ticker | baseline | full | no_technical | no_sentiment | no_fundamental | buy-and-hold |
|---|---|---|---|---|---|---|
| JPM | 0,00 | 8,55 | 4,67 | −0,05 | 4,20 | 32,58 |
| JNJ | 0,00 | −0,52 | −2,01 | −3,30 | −2,77 | −4,55 |
| XOM | 0,00 | −0,72 | 1,89 | −2,61 | 0,08 | −6,38 |
| NVDA | 0,00 | −9,66 | −2,12 | −2,47 | 1,21 | 15,73 |

**Promedio de los 4 tickers por configuración:**

| Configuración | Retorno total | Retorno anualiz. | Volatilidad anual | Sharpe | Calmar | Max DD | Nº operaciones | Coste LLM (USD) |
|---|---|---|---|---|---|---|---|---|
| baseline_price_only | 0,00 % | 0,00 % | 0,0 % | 0,00 | — | 0,00 % | 0,00 | 0,029 |
| full_system | −0,59 % | −0,76 % | 10,5 % | −0,21 | 0,65 | 6,07 % | 12,75 | 0,275 |
| no_technical | 0,60 % | 1,29 % | 10,5 % | −0,23 | 0,46 | 6,67 % | 10,75 | 0,226 |
| no_sentiment | −2,11 % | −4,16 % | 7,7 % | −1,74 | −1,03 | 4,92 % | 9,75 | 0,180 |
| no_fundamental | 0,68 % | 1,42 % | 8,0 % | −0,31 | 0,52 | 4,88 % | 12,50 | 0,176 |
| *buy-and-hold (referencia)* | *9,34 %* | *22,11 %* | *29,1 %* | *0,38* | *2,13* | *14,90 %* | *2* | *0* |

**Deltas globales frente a `full_system`** (pp en retornos y drawdown):

| Configuración | Δ retorno total (pp) | Δ retorno anualiz. (pp) | Δ Sharpe | Δ Calmar | Δ Max DD (pp) | Δ operaciones | Δ coste (USD) |
|---|---|---|---|---|---|---|---|
| baseline_price_only | +0,59 | +0,76 | +0,21 | — | −6,08 | −12,75 | −0,246 |
| no_technical | +1,19 | +2,05 | −0,02 | −0,19 | +0,59 | −2,00 | −0,049 |
| no_sentiment | −1,52 | −3,40 | −1,54 | −1,68 | −1,15 | −3,00 | −0,096 |
| no_fundamental | +1,27 | +2,18 | −0,10 | −0,13 | −1,20 | −0,25 | −0,099 |

**Notas de lectura (observaciones, no interpretación; el análisis es la Fase 5):**

- El baseline (trader solo, sin informes) siempre hace `hold`, así que equivale a mantener el efectivo. Su Sharpe es 0 por convención del motor (volatilidad 0) y su Calmar no está definido: no son comparables con los de las demás configuraciones.
- Las medias de 4 tickers con una sola ejecución son muy sensibles a un solo activo (p. ej. el −9,66 % de `full_system` en NVDA).
- Frente al buy-and-hold, todas las configuraciones LLM quedan por debajo en los dos tickers alcistas (JPM, NVDA) y por encima en los dos bajistas (JNJ, XOM); ninguna supera +0,7 % de retorno medio frente a +9,3 % del buy-and-hold. El patrón encaja con lo descrito por Li et al. (FINSABER).

---

## Desviaciones respecto al cronograma y correcciones a fases previas

| Elemento | Cronograma / fase previa | Realizado |
|---|---|---|
| Infraestructura de 4.1 | Lanzar los backtests y vigilar cuota | Recuperación con replay, logs por día y por llamada, lanzador con tope de gasto |
| Deltas (1.4) | Porcentual en retornos | Absolutos en pp (retornos cerca de 0) |
| Métricas (1.4) | Las de `metrics.json` | Recalculadas sobre la ventana de evaluación; añadido Calmar y nº de operaciones |
| Buy-and-hold (1.1/1.4) | Referencia externa «solo en el análisis (Fase 5)» | Ejecutado ya en 4.3 con el mismo motor (`run_buyhold.py`) |
| Coste (Fase 3.6) | Estimación $1-2, cifras infraestimadas | ≈ 4,4× mayores por el bug de la Responses API; total real $3,54 (corregido en `entregable_fase3.md`) |
| Figuras (5.4) | Gráficos de equity en la Fase 5 | Generados ya en `consolidate_results.py` (curvas normalizadas por ticker) |

---

## Material para la memoria

**Capítulo 4 (Marco Metodológico):** parámetros del experimento; las 5 configuraciones y el buy-and-hold como referencia externa; mecanismo de recuperación y su justificación (D3-D7), aclarando que no se activó; hallazgos técnicos 1-4; definición de métricas (ventana de evaluación, fórmulas, rf, Calmar) y de los deltas y la contribución marginal.

**Capítulo 6 (Limitaciones):** una sola ejecución/semilla y no determinismo del LLM; los informes de analista no se comparten entre configuraciones (D1); 4 tickers y 6 meses en un único régimen de mercado; modelo económico (`gpt-4.1-mini`); baseline degenerado (nunca opera); retornos brutos de coste LLM; la fuente de noticias del rol fundamental está indexada por compañía y se comparte con sentimiento (Fase 2.3); posible leakage residual del modelo no necesariamente uniforme entre configuraciones (Fase 1.2).

**Capítulo 6 (Prospectiva):** más semillas, ventanas y tickers; clasificación de noticias por relevancia con ventanas de distinta amplitud y deduplicación de noticias (Fase 2.2); fuente macro independiente de las compañías.

---

## Artefactos y ubicación

| Artefacto | Ubicación |
|---|---|
| Resultados de FINSABER por par | `C:\finsaber-data\results\<setup>\AblationStrategy\2024-05-13_2025-01-31\<ticker>\` (`equity_curve.csv`, `metrics.json`, `trades.csv`, `rejected_orders.csv`, `llm_costs.csv`) |
| Buy-and-hold | `C:\finsaber-data\results\buy_and_hold\BuyAndHoldIso\2024-08-01_2025-01-31\<ticker>\` |
| Logs de ejecución | `C:\finsaber-data\results\_run_logs\` (`calls.jsonl`, `*.days.jsonl`, `run_all_summary.csv`, `done\`) |
| Figuras de FINSABER | `C:\finsaber-data\results\_figures\` (no existen para JPM) |
| Dataset consolidado y tablas | `Fase4\consolidated\` (`consolidated_results.csv`, `summary_by_config.csv`, `deltas_vs_full_global.csv`, `deltas_vs_full_by_ticker.csv`, `total_return_ticker_x_config.csv`, `figures\equity_<ticker>.png`) |
| Código | `run_all.py`, `run_backtest.py`, `run_buyhold.py`, `consolidate_results.py` (raíz del repo) |
| Backup | copia de `results\` y `_run_logs\` en OneDrive (hecha tras la ejecución) |

## Pendientes abiertos

- Fase 5: análisis cuantitativo, análisis cualitativo (necesita los `*.days.jsonl` de `_run_logs\` para elegir 2-3 episodios), contextualización frente a la literatura y visualizaciones finales.

---

**Entregable Fase 4 — completo:** 20 backtests ejecutados sin fallos, QA superado, dataset consolidado con referencia buy-and-hold, tablas y curvas de equity, y documentación de las decisiones y hallazgos técnicos. Lista para la Fase 5.