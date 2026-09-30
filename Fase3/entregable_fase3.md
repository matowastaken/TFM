# Ficha de implementación — Fase 3 [CERRADA]
*(código fuente en el repo TFM; borrador para Capítulo 4 — Marco Metodológico, y para anexo técnico)*

## 3.1 Estructura del proyecto de código [CERRADO]

- **Repo:** `C:\Users\pablo\TFM`, Git, publicado en GitHub como privado.
- **Entorno:** `.venv-trading` (venv, no conda), Python 3.14.7, pandas 3.0.6.
- **Estructura:**
  ```
  TFM/
  ├── .git/                 # repositorio Git
  ├── .venv-trading/        # entorno virtual del proyecto
  ├── .pytest_cache/        # caché local de pytest
  ├── .env                  # variables locales del entorno
  ├── .gitignore
  ├── agents/               # technical_agent.py, sentiment_agent.py, fundamental_agent.py, trader_agent.py, _common.py
  ├── config/               # ablation.yaml
  ├── llm/                  # call_llm.py
  ├── strategy/             # orchestrator.py, ablation_strategy.py
  ├── prototypes/           # scripts de Fase 2 (technical_prototype.py, etc.)
  ├── tests/                # test_agents.py, test_orchestrator.py
  ├── Fase0/ Fase1/ Fase2/ Fase3/  # entregables y artefactos de fases previas
  ├── cronograma_tfm_ablacion_multiagente.md
  ├── README.md
  ├── requirements.txt
  ├── run_backtest.py       # runner de un backtest para una configuración (Fase 3.5/3.6)
  └── .
  ```
- **`.env`:** `AZURE_OPENAI_KEY`, `AZURE_OPENAI_ENDPOINT`, `GEMINI_API_KEY`, `FINSABER_DATA_PATH=C:\finsaber-data`, `FINSABER_RESULTS_PATH=C:\finsaber-data\results`.
- **`.gitignore`:** `.env`, `*.env`, `.venv-trading/`, `__pycache__/`, `*.pyc`, `.pytest_cache/`, `results/`.
- **`requirements.txt`:** añadidos `pyyaml` y `pytest` a la lista original del cronograma.
- **Nota de nombrado (corrige Fase 0.3, ver 3.5/3.6 más abajo):** los `setup_name` de las 5 configuraciones de ablación **no** llevan el prefijo `cherry_pick_` que se había planeado en la Fase 0/1 — se confirmó en la Fase 3.6, contra el código fuente real, que ese prefijo solo es necesario en el motor `FINSABERBt` (Backtrader), no en `FINSABER` (el motor usado en este proyecto). Los `setup_name` finales son: `full_system`, `no_technical`, `no_sentiment`, `no_fundamental`, `baseline_price_only`.

## 3.2 Wrapper de llamada a LLM [CERRADO]

`llm/call_llm.py` — motor único Azure (`gpt-4.1-mini`, Responses API), sin fallback dinámico a Gemini:

- **Decisión de diseño:** Gemini no se usa como fallback automático ante excepciones. Es una alternativa **manual**, solo si se agota presupuesto/cuota de Azure — para no mezclar proveedores dentro de una misma ejecución y mantener coherencia entre configuraciones de ablación. Mientras tanto, `call_llm()` usa únicamente Azure.
- **Reintentos:** hasta 3 intentos con backoff exponencial (1s/2s/4s) ante `RateLimitError`, `APITimeoutError` o `json.JSONDecodeError`.
- **Fallo persistente:** tras 3 intentos, `call_llm()` **no** lanza excepción — loguea el error y devuelve `None`. El día/ticker correspondiente se marca como fallido en el nivel que lo consume (agente → orquestador → estrategia), en vez de detener toda la ejecución y arriesgar un gasto de tokens descontrolado ante un fallo generalizado.
- **Bug real de coste (corrige el hallazgo de la Fase 0.4):** parchear `OPENAI_PRICE["gpt-4.1-mini"] = {"input": 0.40, "output": 1.60}` no bastaba. `_openai_cost()`, en `backtest.toolkit.llm_cost_monitor`, comprueba subcadenas literales hardcodeadas (`"gpt-4o-mini"`, `"gpt-4o"`) en vez de buscar el modelo por clave en `OPENAI_PRICE`, y `"gpt-4.1-mini"` no coincide con ninguna — el coste seguía registrándose como `0.0` incluso con el parche. Verificado en vivo contra el paquete instalado (`finsaber==2.0.1`). Solución real: se reemplaza `_openai_cost()` por una función que busca el modelo por clave exacta en `OPENAI_PRICE`.
- **Segundo bug de coste (Responses API, detectado en la Fase 4):** además de lo anterior, `add_openai_cost_from_response()` lee `usage.prompt_tokens` (nombre de la Chat Completions API), que no existe en la Responses API (donde es `input_tokens`). El coste de entrada se contabilizaba como 0 y el coste total quedaba infraestimado (~4,4×). Solución: `call_llm._record_cost()` llama a `add_llm_cost(model=AZURE_MODEL, prompt_tokens=input_tokens, completion_tokens=output_tokens, provider="openai")`. **Todas las cifras de coste de la Fase 3.6 reportadas antes de este parche están infraestimadas** (ver correcciones en 3.6).
- **Precio confirmado** (búsqueda directa, Azure AI Foundry, Global Standard): `gpt-4.1-mini` = $0.40 / 1M tokens input, $1.60 / 1M tokens output.

## 3.3 Los 4 agentes [CERRADO]

`agents/_common.py` (validación de los dos esquemas de salida + `select_news()`, compartida por sentimiento y fundamental) y un módulo por rol: `technical_agent.py`, `sentiment_agent.py`, `fundamental_agent.py`, `trader_agent.py`.

- **Firma:** parámetros explícitos y tipados por agente (no un dict `context` genérico), p. ej. `technical_agent.run(date, ticker, ohlcv_history)`.
- **Cálculo de indicadores y selección de noticias** viven dentro de cada agente, no en un módulo de preparación de datos aparte.
- **`technical_agent.py`:** indicadores en pandas puro (SMA20/50, RSI(14) Wilder, MACD(12,26,9), Bollinger(20,2), ATR(14), volumen relativo 20d), todos causales. Bug corregido: cuando `avg_loss=0` (solo subidas en la ventana), la fórmula de RSI daba `NaN`; ahora devuelve `RSI=100` explícitamente.
- **`trader_agent.py`:** agregación por criterio propio del LLM, no mecánica. La traducción `size → quantity` **no** vive aquí — queda pospuesta a `strategy/` (Fase 3.5), para no pedirle al LLM que calcule cash o número de acciones.
- **Fallos:** si `call_llm()` devuelve `None`, o si la respuesta no valida contra el esquema esperado, el `run()` del agente devuelve `None` (con `logger.warning` incluyendo la respuesta cruda cuando el fallo es de validación, no de red). Si el trader responde con esquema válido pero un `role` de analista incorrecto, se descarta en silencio.
- **Bug real corregido en producción (encontrado en el smoke test de 3.6, no en el diseño inicial):** el prompt del trader indica que `size` se ignora si la acción es `hold`, y el modelo a veces **omite la clave por completo** en ese caso (no la envía como `null`, la omite). La validación original lo rechazaba como esquema inválido. Fix: un `hold` sin `size` se normaliza a `0.0` antes de validar; `buy`/`sell` siguen exigiendo `size`.
- **Tests:** `tests/test_agents.py`, 16 tests (los 12 originales + 2 para el caso `hold` sin `size`), todos con `call_llm` mockeado. 16/16 passed en el entorno real de Pablo (Python 3.14.7, pytest 9.1.1).

## 3.4 Orquestador de configuración [CERRADO]

`config/ablation.yaml` (5 configuraciones) + `strategy/orchestrator.py` (`run_configuration()`, `load_ablation_config()`).

- **Decisión clave:** si un analista activo falla (`run()` devuelve `None`), **todo el día/ticker se marca como fallido** en esa configuración — el trader no se llama con un subconjunto parcial de informes. Llamarlo con menos informes de los que corresponden a la configuración activa contaminaría la comparación de ablación: un día con un fallo técnico puntual en `full_system` se volvería indistinguible, a nivel de input del trader, de un día normal en `no_technical`.
- `OrchestratorResult` conserva los informes obtenidos hasta el punto del fallo (no los descarta), para dar contexto en el QA de la Fase 4.2 sin tener que re-ejecutar nada.
- **Tests:** `tests/test_orchestrator.py`, 5/5 passed.

## 3.5 Interfaz de estrategia FINSABER-2 [CERRADO]

`strategy/ablation_strategy.py` — `AblationStrategy(BaseStrategyIso)`.

Puntos confirmados contra el código fuente real de `finsaber==2.0.1` instalado (no solo documentación), que obligaron a rediseñar partes del plan original:

- `on_data(date, today_data, framework)` se llama **una vez por fecha**, no por ticker; `today_data` solo trae los datos de ese día (`today_data["price"][ticker]`, `today_data["news"][ticker]` = lista de textos), sin histórico. La estrategia mantiene su propio buffer acumulado de OHLCV y noticias, construido incrementalmente en cada `on_data()`.
- `date` llega como `datetime.date`, no `pd.Timestamp` — se normaliza al entrar en `on_data()`.
- `today_data["price"][ticker]` trae campos crudos (`open/high/low/close`) y ajustados por splits/dividendos (`adjusted_open/adjusted_high/adjusted_low/adjusted_close`). La ejecución real de órdenes usa los **ajustados** (fill a `next_open` contra `adjusted_open`), así que los indicadores técnicos y el precio de referencia pasado a `framework.buy()`/`.sell()` también usan los campos ajustados, por consistencia.
- El `price` que se pasa a `framework.buy(date, ticker, price, quantity)` es solo un `reference_price` (fallback) — la ejecución real resuelve el precio de fill internamente contra el bar del día de ejecución. Confirma que el diseño de la traducción `size → quantity` (Fase 2.4) era correcto sin cambios.
- **Diseño de warmup (rediseñado a partir de feedback):** en vez de dejar al analista técnico sin operar durante los primeros ~50 días reales del periodo de test, el runner de 3.6 carga el dataset ~50 sesiones **antes** de la fecha real de evaluación. `AblationStrategy` recibe un `evaluation_start_date` explícito: acumula buffers desde el primer día cargado, pero no opera (ni llama al orquestador) hasta esa fecha — así el primer día real de evaluación ya tiene indicadores técnicos completos, sin "gastar" el periodo de test en calentamiento.
- Una instancia de `AblationStrategy` por (`setup_name`, `ticker`) — coherente con el diseño de 1.3 (20 backtests = 5 configuraciones × 4 tickers, capital independiente por ticker).
- `_position_state()` se lee de `framework.portfolio`, no se inventa. `_execute_decision()` traduce `size → quantity` en Python puro, tal como se decidió en 2.4.
- Verificado end-to-end en un entorno de pruebas (FINSABERFrameworkHelper + AblationStrategy reales, dataset sintético, `trader_agent.run` mockeado): 18 trades ejecutados, 0 `rejected_orders`, métricas coherentes.

## 3.6 Smoke test [CERRADO]

`run_backtest.py` — runner de un backtest para una configuración de ablación, conectando `FinsaberParquetDataset` (`FINSABER_DATA_PATH`) y `log_base_dir` (`FINSABER_RESULTS_PATH`).

**API real de `FINSABER` confirmada** (distinta de lo asumido inicialmente por el cronograma): no se instancia la estrategia a mano — se llama a `FINSABER(trade_config).run_iterative_tickers(AblationStrategy, strat_params, tickers)`, y el framework instancia la clase por nosotros, resolviendo claves dinámicas (`"$symbol"`, `"$date_from"`, `"$date_to"`) en `strat_params` contra el propio `trade_config`. Encaja con la firma de `AblationStrategy.__init__(setup_name, active_roles, ticker, evaluation_start_date)`.

**Tres bugs adicionales de `finsaber==2.0.1` encontrados y resueltos en esta fase** (todos verificados en vivo, no solo por lectura de código):

1. **Persistencia `cherry_pick_`** (corrige el hallazgo de la Fase 0.3): el check `"cherry_pick" in setup_name` que activa `write_result_artifacts()` vive únicamente en `finsaber_bt.py` (motor `FINSABERBt`, Backtrader) — no en `finsaber.py` (motor `FINSABER`, el usado en este proyecto según la Fase 0.4). Con `FINSABER`, la persistencia es incondicional (`if trade_config.save_results`, por defecto `True`). El prefijo se elimina de los `setup_name` en todo el proyecto.
2. **`aggregate_results_one_strategy()` (paso final, opcional, de `run_iterative_tickers()`):** (a) se llama sin pasar `output_dir`, así que cae a una ruta relativa por defecto dentro del propio paquete instalado en vez de usar `log_base_dir`, y crashea con `FileNotFoundError`; (b) usa `DataFrame._append()`, un método privado de pandas eliminado en pandas ≥ 2.x (confirmado con pandas 3.0.6). Esta función solo consolida resultados entre *rolling windows*, algo que este proyecto no usa (Fase 1.2: un único periodo de test) — la consolidación real entre las 5 configuraciones × 4 tickers se hace a mano en la Fase 4.3 a partir de los artefactos que sí se escriben correctamente. Se desactiva sustituyéndola por un no-op en `run_backtest.py`.
3. **Estructura de retorno:** `run_iterative_tickers()` con `save_results=True` devuelve `{periodo: {ticker: metrics}}`, no `{ticker: metrics}` directamente.

**Ejecuciones reales confirmadas** (NVDA, evaluación 2024-08-01 → 2024-08-08, preload desde 2024-05-13, Azure real):

- `baseline_price_only`: 6 llamadas (una por día), todas `200 OK`, `total_llm_cost` reportado = $0.0005376 (**infraestimado**: solo contabilizaba tokens de salida, ver 3.2; coste real no remedido con el parche), 0 `rejected_orders`, 0 trades (el trader decidió `hold` los 6 días, comportamiento esperado sin informes de analistas).
- `full_system` (antes del fix del bug de `size` ausente): 24 llamadas, 2 trades ejecutados, `total_llm_cost` reportado ≈ $0.0035 (**infraestimado**, solo salida); tras el parche, una ejecución equivalente de 24 llamadas midió ≈ $0.0152 (≈ $0.00063/llamada), pero 2-3 de 6 días marcados como fallidos por el trader pese a `HTTP 200` en todas las llamadas.
  - **Causa confirmada:** el prompt del trader indica que `size` se ignora si la acción es `hold`, y el modelo a veces omite esa clave — la validación original lo rechazaba. Corregido en 3.3 (ver arriba). Efecto sobre los resultados de trading: nulo (un día fallido no genera ninguna orden, igual que un `hold`); el daño real era solo ruido en el recuento de fallos del QA de la Fase 4.2.
  - **Re-ejecución tras el fix:** 6/6 días procesados sin ningún fallo.
- **No determinismo observado** entre ejecuciones idénticas (`temperature=0.2`): la primera ejecución de `full_system` compró con `size=0.40` (retorno −4.28%), la segunda con `size=0.50` (retorno −5.36%), con el mismo patrón cualitativo de decisiones (compra el 08-01, venta total el 08-02, hold el resto). Reafirma la limitación de una sola ejecución/semilla, ya prevista para el Capítulo 6 (Prospectiva).
- El precio de ejecución respeta `execution_timing="next_open"`: primera operación con `signal_date=2024-08-01`, `execution_date=2024-08-02` (confirmado en `trades.csv`).

**Nota para la Fase 4.3 (consolidación):** `metrics.json` mezcla unidades — `total_return`/`annual_return`/`annual_volatility` en fracción (p. ej. `-0.0428`), pero `max_drawdown` en porcentaje positivo (`5.4958` = 5.50%). Además no incluye Calmar ratio ni número de operaciones (que sí pide la Fase 1.4) — Calmar se calcula a mano con unidades consistentes, y el número de operaciones se cuenta desde `trades.csv`.

**Estimación de coste para la Fase 4** (corregida tras el parche de la Responses API, ver 3.2): el coste real medido es ≈ $0.00063/llamada de `full_system` (los roles de noticias, sentimiento y fundamental, consumen ≈ 1.900 tokens de entrada; técnico y trader ≈ 550). Con ~125 días de trading por ticker (agosto 2024 – enero 2025) y las 5 configuraciones (4+3+3+3+1 = 14 llamadas/día en total entre las 5), el experimento completo (20 backtests, 4 tickers, ≈ 7.000 llamadas) se estima en torno a **$4** en total — muy por debajo del crédito disponible de Azure. *(La estimación inicial de $1-2, basada en ≈ $0.000144/llamada, estaba infraestimada por el bug de coste.)*

---

**Entregable Fase 3 — completo:** repositorio funcional (`agents/`, `llm/`, `strategy/`, `config/`, `run_backtest.py`, `tests/`), con los 4 agentes, el orquestador y la estrategia FINSABER-2 verificados de extremo a extremo contra Azure real, un smoke test limpio (0 días fallidos, 0 `rejected_orders`) sobre NVDA, y un conjunto de hallazgos de compatibilidad con `finsaber==2.0.1` (documentados arriba) que corrigen y completan las decisiones de las Fases 0-2. Lista para la Fase 4 (ejecución completa del experimento).
