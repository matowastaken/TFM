# Cronograma detallado — Estudio de ablación de un sistema multiagente LLM de trading

**Proyecto:** Ablación por rol de un sistema multiagente (4 roles: técnico, sentimiento, fundamental macro/geopolítico, trader) inspirado en TradingAgents, evaluado sobre el framework FINSABER-2.

**Recursos:** Azure AI Foundry (`gpt-4.1-mini`, crédito estudiante) como motor principal, Gemini API free tier (`gemini-3.5-flash-lite`) como fallback puntual y validación cruzada, FINSABER-2 como harness de backtesting.

**Duración estimada total:** 9-10 semanas a tiempo parcial (~10-12h/semana) ≈ 90-110 horas de trabajo efectivo.

---

## FASE 0 — Validación de viabilidad (previa a comprometer el diseño) ✅ COMPLETADA
*Duración: 3-5 horas | Semana 1*

### 0.1 Confirmar acceso y condiciones de Azure ✅
- Recurso creado: **Azure AI Foundry** (no el recurso "Azure OpenAI" clásico).
- Modelo desplegado: **`gpt-4.1-mini`** (Standard/Global Standard, sin servidor). Se descartó `gpt-4o-mini` por estar deprecated y sin cuota disponible.
- SDK y API confirmados con código real:
  - Librería: `openai.OpenAI` (**NO** `AzureOpenAI`).
  - Endpoint: `https://foundry-tfm-trading.services.ai.azure.com/openai/v1`.
  - Método: `client.responses.create()` (Responses API), **no** `chat.completions.create()`.
  - Extracción de texto: `response.output_text`.
- Cuota confirmada: **100.000 TPM / 100 RPM**.
- Configuración de red: "Todas las redes, incluso Internet" (adecuado para desarrollo individual; la seguridad recae en la API key, no en restricción de red).
- Fecha de caducidad del crédito: confirmada, suficiente para finalizar el TFM.

### 0.2 Confirmar acceso a Gemini API free tier ✅
- API real utilizada: **Interactions API** (`client.interactions.create()`), que sustituyó a la antigua `generate_content` durante el desarrollo de esta fase.
- Modelo final elegido: **`gemini-3.5-flash-lite`** (RPM: 15 | TPM: 250.000 | RPD: 500). Se descartó `gemini-2.0-flash` (retirado) y `gemini-3.6-flash` estándar (RPD=20, insuficiente).
- Extracción de texto: `interaction.output_text`.
- **Decisión de diseño**: Gemini se reserva como **fallback puntual** (errores/timeouts de Azure) y para **validación cruzada de formato** (Fase 2.5) — **no** como motor de ningún rol, por la fragilidad observada del tier gratuito (dos cambios de modelo/API en una misma sesión de trabajo).

### 0.3 Instalar y probar FINSABER-2 ✅
- Instalación: `pip install finsaber` dentro de un entorno virtual (`venv`) dedicado — **importante**: crear el venv fuera de carpetas sincronizadas (OneDrive/Drive) para evitar bloqueos de Windows Controlled Folder Access y problemas de sincronización con miles de ficheros.
- Dataset: **`waylonli/FINSABER-V2-Data`** en Hugging Face (~7.5 GB, 97 ficheros parquet), descargado con `huggingface_hub.snapshot_download()`. Guardar en ruta **local no sincronizada** (ej. `C:\finsaber-data`), nunca dentro de OneDrive/Documents si Controlled Folder Access está activo.
- Cobertura confirmada del dataset:
  - `price_daily`: 2000-2025 (todo S&P 500).
  - `news_items`: **solo 2009-2025** (restricción clave para la Fase 1.2: el periodo de test debe estar dentro de este rango).
  - `filingk` / `filingq` (10-K anual / 10-Q trimestral): 2000-2025.
- Quick Start ejecutado con éxito (`total_return = 0.00443` sobre AAPL, Buy-and-Hold, 2024-01-02 a 2024-01-10).
- **⚠️ Hallazgo importante sobre persistencia de resultados**: `run_iterative_tickers()` (método usado en agentes Python-nativos) solo invoca `write_result_artifacts()` si `setup_name` contiene la cadena `"cherry_pick"` (particularidad interna ligada a los experimentos del paper original, no documentada).
  - **Decisión de diseño**: en vez de modificar el código fuente del framework, todos los `setup_name` de la Fase 1.3 llevarán el prefijo **`cherry_pick_`** (ej. `cherry_pick_full_system`, `cherry_pick_no_technical`, etc.) para activar la persistencia nativa. Documentar esta convención explícitamente en el Capítulo 4.
  - Ruta de artefactos: `log_base_dir/setup_name/strategy_name/date_from_date_to/ticker/`, con `equity_curve.csv`, `metrics.json`, `orders.csv`, `trades.csv`, `rejected_orders.csv`, `llm_costs.csv`.
  - `log_base_dir` por defecto apunta dentro de `site-packages` del venv — **no ideal**; especificar explícitamente una ruta propia (ej. `C:\finsaber-data\results\`) en la Fase 3.

### 0.4 Revisar la interfaz de estrategia "Python-native agent / LLM loop" ✅
- Validado con **código fuente real** (no solo documentación), que reveló discrepancias respecto a la doc web.
- Motor a usar: **`FINSABER`** con la clase base **`BaseStrategyIso`** (`finsaber.strategy.timing_llm`).
- Método a implementar: `on_data(self, date, today_data, framework)`, donde `today_data` es el diccionario de datos de ese día concreto (vía `data_loader.get_data_by_date(date)`, invocado internamente por `FINSABERFrameworkHelper.run()`).
- Métodos disponibles en `framework` (`FINSABERFrameworkHelper`):
  - `framework.buy(date, ticker, price, quantity)` / `framework.sell(...)`.
  - `quantity=-1` → usa todo el cash disponible (buy) o vende toda la posición (sell). **Cuidado en multi-ticker**: no usar `-1` a ciegas si se opera más de un ticker simultáneamente, para repartir capital correctamente.
  - Gestión automática de comisión, slippage, liquidity cap y rechazo de órdenes (fondos insuficientes, histórico de liquidez insuficiente) — no hay que reimplementar esta lógica.
- **LLM cost accounting — NO es automático**:
  - Debe invocarse explícitamente desde el wrapper `call_llm()`: `add_openai_cost_from_response(response)` (o `add_llm_cost(...)` con tokens manuales), del módulo `backtest.toolkit.llm_cost_monitor`.
  - **⚠️ Bug/limitación detectada**: el diccionario `OPENAI_PRICE` del monitor solo incluye precios para `gpt-4o-mini` y `gpt-4o` — **no incluye `gpt-4.1-mini`**, el modelo real del proyecto. Si no se corrige, el coste se registraría siempre como `0.0`.
  - **Acción obligatoria para la Fase 3.2**: parchear `OPENAI_PRICE` añadiendo la entrada de `gpt-4.1-mini` con precios reales antes de usar el monitor, ej.:
    ```python
    from backtest.toolkit.llm_cost_monitor import OPENAI_PRICE
    OPENAI_PRICE["gpt-4.1-mini"] = {"input": 0.40, "output": 1.60}  # verificar precios vigentes
    ```
  - El monitor solo soporta `provider="openai"` — el coste de las llamadas a Gemini (fallback) no se registra automáticamente; si se desea trazabilidad completa, implementar conteo manual para esas llamadas puntuales.

**Entregable Fase 0 (completado):** documento de viabilidad confirmando acceso funcional a Azure AI Foundry, Gemini, y FINSABER-2; cobertura de datos verificada; interfaz de estrategia y mecanismo de coste LLM entendidos a nivel de código fuente, con 2 hallazgos críticos documentados (convención `cherry_pick_` y parche de precios) que condicionan el diseño de las Fases 1 y 3.

---

## FASE 1 — Diseño experimental cerrado
*Duración: 6-8 horas | Semana 1-2*

### 1.1 Definir el objetivo de investigación específico y su vínculo con los objetivos del TFM
- Redactar la pregunta de investigación concreta (ej. "¿Qué contribución marginal aporta cada rol especializado a la performance ajustada a riesgo de un sistema multiagente de trading?").
- Vincular explícitamente esta pregunta con el objetivo específico correspondiente del Capítulo 2.

### 1.2 Selección de tickers y periodo de test
- Elegir 3-5 tickers con distintos perfiles (ej. 1 tech de alta volatilidad, 1 valor defensivo, 1 con alta exposición a noticias macro).
- Elegir un periodo de test corto pero real (2-3 meses), fuera del rango de entrenamiento de los LLM usados (evitar contaminación/leakage de conocimiento del modelo) **y dentro del rango 2009-2025**, ya que el dataset de noticias (`news_items`) de FINSABER-V2-Data no cubre años anteriores a 2009 (confirmado en Fase 0.3).
- Confirmar disponibilidad de datos de precio + noticias para ese periodo y tickers en el dataset descargado.

### 1.3 Definir las 5 configuraciones de ablación
1. Sistema completo (4 roles).
2. Sin analista técnico.
3. Sin analista de sentimiento.
4. Sin analista fundamental macro/geopolítico.
5. Baseline: solo trader con precio crudo (sin analistas).

**Convención de nombrado obligatoria** (por el hallazgo de la Fase 0.3): todos los `setup_name` deben llevar el prefijo `cherry_pick_` para activar la persistencia nativa de resultados de FINSABER-2, ej.:
- `cherry_pick_full_system`
- `cherry_pick_no_technical`
- `cherry_pick_no_sentiment`
- `cherry_pick_no_macro`
- `cherry_pick_baseline_price_only`

### 1.4 Definir métricas de evaluación
- Retorno total y anualizado, Sharpe, Max Drawdown, Calmar, nº de operaciones, y coste total de LLM por configuración (usando el monitor de coste ya parcheado en Fase 0.4).
- Definir explícitamente cómo se comparará cada configuración contra el sistema completo (delta de cada métrica).

### 1.5 Fuente de noticias para sentimiento y macro/geopolítico
- Fuente confirmada: dataset `news_items` de FINSABER-V2-Data (cobertura 2009-2025), ya descargado y validado en Fase 0.3.
- Si el periodo/tickers elegidos requieren más densidad de noticias de la disponible, evaluar fuente externa complementaria (ej. NewsAPI free tier) y diseñar script de descarga/almacenamiento local.

**Entregable Fase 1:** ficha de diseño experimental cerrada (1-2 páginas) que podrás pegar casi literalmente en el Capítulo 4 (Marco Metodológico).

---

## FASE 2 — Diseño de los 4 roles (prompts e interfaces)
*Duración: 10-14 horas | Semana 2-3*

### 2.1 Rol: Analista técnico
- Definir inputs (OHLC, indicadores: medias móviles, RSI, MACD, etc. — decidir cuáles calcular y con qué librería, ej. `ta` o `pandas-ta`).
- Diseñar prompt con formato de salida estructurado (JSON: señal + justificación breve).
- Prototipo rápido con 2-3 ejemplos manuales para validar formato de salida, usando `client.responses.create()` contra el deployment `gpt-4.1-mini` (Fase 0.1).

### 2.2 Rol: Analista de sentimiento
- Definir inputs (titulares/noticias del día o ventana reciente sobre el ticker, extraídos de `news_items`).
- Diseñar prompt de resumen + puntuación de sentimiento (escala simple, ej. -1 a 1) + justificación.
- Prototipo rápido de validación.

### 2.3 Rol: Analista fundamental macro/geopolítico
- Definir inputs (noticias macro generales: tipos de interés, inflación, eventos geopolíticos relevantes al periodo — no específicas del ticker; opcionalmente enriquecer con fragmentos de `filingk`/`filingq` si aporta valor).
- Diseñar prompt que evalúe el contexto macro y su posible impacto direccional sobre el activo/sector.
- Prototipo rápido de validación.

### 2.4 Rol: Trader / decisor final
- Diseñar prompt que agregue las salidas de los roles activos en cada configuración (nota: debe ser capaz de funcionar con 1, 2, 3 o 4 inputs, según la configuración de ablación) y devuelva la orden final (buy/hold/sell + tamaño de posición si aplica).
- Definir lógica de agregación cuando faltan roles (ej. en la configuración "sin sentimiento", el trader simplemente no recibe ese input, sin necesidad de rediseñar el prompt desde cero).
- Tener en cuenta el manejo de `quantity=-1` en `framework.buy()`/`.sell()` (Fase 0.4): decidir tamaño de posición explícito si se opera con varios tickers en paralelo.

### 2.5 Validación cruzada de formato
- Probar cada prompt con Azure (`gpt-4.1-mini`, vía Responses API) y con Gemini (`gemini-3.5-flash-lite`, vía Interactions API), comparar consistencia de formato y calidad de output.
- Confirmar el comportamiento de fallback: Azure como motor único de los 4 roles; Gemini utilizado aquí solo para contraste de formato y como red de seguridad ante fallos de Azure (no como motor de producción de ningún rol — decisión tomada en Fase 0.2).
- Documentar decisiones de asignación para justificarlas en la memoria.

**Entregable Fase 2:** especificación de los 4 prompts (versión final), con ejemplos de input/output documentados — material directo para un anexo técnico.

---

## FASE 3 — Implementación del sistema
*Duración: 18-24 horas | Semana 3-5*

### 3.1 Estructura del proyecto de código
- Crear repositorio/estructura de carpetas: `agents/`, `strategy/`, `config/`, `data/`, `results/`.
- Entorno virtual (`venv`, no conda) ya creado en Fase 0 — reutilizar, situado **fuera de carpetas sincronizadas** (OneDrive/Drive).
- `requirements.txt` generado con `pip freeze` a partir del entorno validado en Fase 0 (`openai`, `google-genai`, `python-dotenv`, `finsaber`, `huggingface_hub`).
- Variables de entorno vía `.env` + `python-dotenv` (ya configurado en Fase 0), incluyendo `AZURE_OPENAI_KEY`, `AZURE_OPENAI_ENDPOINT`, `GEMINI_API_KEY`, `FINSABER_DATA_PATH`. `.env` y el propio entorno virtual excluidos vía `.gitignore`.

### 3.2 Implementar wrapper de llamada a LLM
- Función única `call_llm(provider, prompt, schema)` que abstraiga Azure (`client.responses.create()`) vs Gemini (`client.interactions.create()`) — formatos de entrada/salida distintos entre ambos SDKs (Fase 0.1/0.2), normalizar en el wrapper.
- Manejo de reintentos y errores (rate limit, timeout, respuesta mal formada) con **fallback automático a Gemini** si Azure falla repetidamente (rol de Gemini confirmado en Fase 0.2).
- Logging de coste/tokens por llamada usando el LLM cost accounting de FINSABER (`backtest.toolkit.llm_cost_monitor`):
  - **Paso obligatorio previo**: parchear `OPENAI_PRICE` con la entrada de `gpt-4.1-mini` (hallazgo Fase 0.4) antes de instrumentar las llamadas, o el coste se registrará como 0.
  - Llamar a `add_openai_cost_from_response(response)` tras cada llamada a Azure.
  - Para llamadas de fallback a Gemini, implementar conteo de coste manual si se desea trazabilidad completa (el monitor nativo no soporta `provider="gemini"`).

### 3.3 Implementar cada agente como módulo independiente
- `technical_agent.py`, `sentiment_agent.py`, `macro_agent.py`, `trader_agent.py`.
- Cada uno con función `run(date, ticker, context) -> output_estructurado`.
- Tests unitarios mínimos (2-3 casos por agente) para verificar parsing robusto del output.

### 3.4 Implementar el orquestador de configuración
- Función que, dado un conjunto de roles activos (según la configuración de ablación), ejecute únicamente esos agentes y pase sus outputs al trader.
- Parametrizar las 5 configuraciones desde un único archivo de config (YAML/JSON), usando los `setup_name` con prefijo `cherry_pick_` definidos en Fase 1.3.

### 3.5 Implementar la interfaz de estrategia requerida por FINSABER-2
- Heredar de `BaseStrategyIso` (`finsaber.strategy.timing_llm`) e implementar `on_data(self, date, today_data, framework)`, según validado con código real en Fase 0.4.
- Usar el motor `FINSABER` (no `FINSABERBt`, reservado para estrategias tipo Backtrader/técnicas clásicas).
- Conectar la fuente de datos: `FinsaberParquetDataset` apuntando a `FINSABER_DATA_PATH` (precio, noticias, filings) según lo validado en Fase 0.3.
- Especificar explícitamente `log_base_dir` en la configuración, apuntando fuera de `site-packages` (ej. `C:\finsaber-data\results\`), para no depender de la ruta por defecto interna al venv (hallazgo Fase 0.3).

### 3.6 Prueba de humo (smoke test) a pequeña escala
- Ejecutar la configuración completa sobre 1 ticker y 3-5 días.
- Verificar que el pipeline corre de principio a fin sin errores y que FINSABER genera resultados/artefactos correctamente en la ruta esperada (`log_base_dir/cherry_pick_.../strategy/fechas/ticker/`).
- Confirmar que `metrics.json` refleja un `total_llm_cost` distinto de 0 (verificación de que el parche de `OPENAI_PRICE` funciona correctamente).
- Corregir errores de integración antes de pasar a la ejecución completa.

**Entregable Fase 3:** código funcional y probado a pequeña escala, listo para la ejecución completa del experimento.

---

## FASE 4 — Ejecución del experimento
*Duración: 8-12 horas de trabajo activo (más tiempo de espera de ejecución en background) | Semana 5-6*

### 4.1 Ejecución de las 5 configuraciones
- Lanzar, para cada una de las 5 configuraciones (`cherry_pick_...`), el backtest sobre los 3-5 tickers y el periodo definido.
- Monitorizar consumo de crédito Azure (100.000 TPM / 100 RPM, Fase 0.1) y cuota Gemini (15 RPM / 250.000 TPM / 500 RPD, Fase 0.2) durante la ejecución; pausar/ajustar si se acerca a límites — el volumen proyectado del experimento está cómodamente dentro de estos límites según lo estimado en Fase 0.

### 4.2 Control de calidad de la ejecución
- Revisar logs en busca de fallos de parsing, llamadas fallidas o inputs vacíos (ej. días sin noticias).
- Re-ejecutar puntualmente los días/tickers con errores detectados.

### 4.3 Consolidación de resultados
- Extraer de FINSABER los artefactos estructurados de resultados (`equity_curve.csv`, `metrics.json`, `llm_costs.csv`) de la ruta `log_base_dir` configurada en Fase 3.5, para cada una de las 5 configuraciones.
- Consolidar en una tabla comparativa única (configuración × métrica).

**Entregable Fase 4:** dataset de resultados consolidado (tabla + curvas de equity) para las 5 configuraciones.

---

## FASE 5 — Análisis de resultados
*Duración: 10-14 horas | Semana 6-7*

### 5.1 Análisis cuantitativo de la ablación
- Calcular la contribución marginal de cada rol (delta de métricas respecto al sistema completo al eliminarlo).
- Identificar si algún rol introduce ruido (empeora el sistema al estar presente) — hallazgo potencialmente relevante.

### 5.2 Análisis cualitativo de casos
- Seleccionar 2-3 episodios concretos (ej. un día de alta volatilidad o noticia relevante) y examinar el razonamiento de cada agente para ilustrar cómo influye su ausencia/presencia en la decisión final.

### 5.3 Contextualización frente a la literatura
- Comparar tus hallazgos con lo reportado (o no reportado) en TradingAgents, FinCon, FinMem respecto a la contribución de roles.
- Conectar con las limitaciones metodológicas discutidas en tu Sección 3.4 (p. ej., ausencia de ablaciones en los papers originales).

### 5.4 Visualizaciones
- Gráficos de equity curve por configuración.
- Gráfico de barras de contribución marginal por rol y métrica.

**Entregable Fase 5:** borrador completo del Capítulo 5 (Presentación de análisis, resultados y discusión), con tablas y gráficos.

---

## FASE 6 — Redacción e integración en el TFM
*Duración: 14-18 horas | Semana 7-9*

### 6.1 Capítulo 4 — Marco Metodológico
- Volcar la ficha de diseño experimental (Fase 1) en las secciones correspondientes: población/muestra (tickers y periodo), objetivos de investigación, variables e instrumentos (los 4 roles y sus prompts, resumidos), diseño experimental (las 5 configuraciones), modelo de análisis de datos (métricas y comparación).
- Documentar explícitamente las decisiones de infraestructura surgidas en la Fase 0, relevantes para la reproducibilidad:
  - Uso de Azure AI Foundry (`gpt-4.1-mini`) como motor único de los 4 roles, con Gemini (`gemini-3.5-flash-lite`) como mecanismo de fallback y validación cruzada — justificar por estabilidad de cuota frente a la fragilidad observada del tier gratuito de Gemini.
  - Convención de nombrado `cherry_pick_` en los `setup_name`, motivada por una particularidad no documentada de la persistencia de resultados en FINSABER-2 (`run_iterative_tickers`), y la decisión de no modificar el código fuente del framework.
  - Parche aplicado a `OPENAI_PRICE` en el monitor de coste de FINSABER-2 para soportar `gpt-4.1-mini`.

### 6.2 Capítulo 5 — Resultados y discusión
- Redactar a partir del borrador de la Fase 5.4-5.3, siguiendo el estilo impersonal y el formato de la plantilla UAX.

### 6.3 Ajuste del Capítulo 3 (Marco Teórico) — cierre del bridge
- Revisar que la Sección 3.4 (limitaciones metodológicas) referencia explícitamente la ausencia de ablaciones por rol en los sistemas revisados, preparando al lector para tu contribución en el Capítulo 5.

### 6.4 Capítulo 6 — Conclusiones
- Redactar limitaciones específicas del estudio (tamaño de muestra reducido, periodo corto, uso de LLMs económicos que pueden diferir en calidad de GPT-4/Claude usados en los papers originales, ausencia de múltiples semillas/repeticiones por coste, cobertura de noticias limitada a 2009-2025 en el dataset usado).
- Redactar prospectiva (ampliar periodo, más tickers, más semillas, roles adicionales).

### 6.5 Anexos técnicos
- Incluir especificación de prompts (Fase 2.5), fragmentos de código relevantes (incluyendo el wrapper `call_llm()` y el parche de `OPENAI_PRICE`), tabla completa de resultados.

**Entregable Fase 6:** TFM con Capítulos 3-6 integrados y coherentes con el hilo narrativo de validación rigurosa.

---

## FASE 7 — Margen de contingencia y revisión final
*Duración: 6-10 horas | Semana 9-10*

### 7.1 Buffer para imprevistos técnicos
- Reservar tiempo explícito para problemas no anticipados (cambios en la API de FINSABER, rate limits inesperados, errores de formato persistentes en algún rol, posibles cambios adicionales en las APIs de Azure/Gemini como los observados durante la Fase 0).

### 7.2 Revisión de coherencia global
- Releer el TFM completo verificando que el hilo "validación rigurosa, no innovación" se mantiene desde la introducción hasta las conclusiones.

### 7.3 Revisión de formato UAX
- Verificar estilos (Título Capítulo/Apartado/Subapartado, Texto TFM, Bibliografía APA con sangría francesa), índice actualizado, extensión total dentro del rango 50000-60000 palabras.

---

## Resumen de dependencias críticas

| Fase | Depende de | Riesgo si falla |
|---|---|---|
| 0 (viabilidad) | — | ✅ Completada — 2 hallazgos críticos incorporados al diseño |
| 1 (diseño) | 0 (viabilidad confirmada) | Rediseñar tickers/periodo si faltan datos; periodo debe respetar cobertura de noticias 2009-2025 |
| 2 (prompts) | 1 (configuraciones definidas) | Bajo — iterable |
| 3 (implementación) | 2 (prompts validados) | Medio — mitigado por el conocimiento detallado de la interfaz de FINSABER-2 obtenido en Fase 0.4 |
| 4 (ejecución) | 3 (smoke test superado) | Alto — si el smoke test falla, no lanzar la ejecución completa todavía |
| 5 (análisis) | 4 (resultados consolidados) | Bajo |
| 6 (redacción) | 5 (borrador de resultados) | Bajo — puede empezar en paralelo con 6.1/6.3 |

**Recomendación:** no comprimir el smoke test de la Fase 3.6 — es la fase más barata en horas y la que más tiempo total ahorra si detecta problemas pronto (como ya demostró la propia Fase 0 al descubrir la convención `cherry_pick_` y el bug de precios antes de la implementación completa).
