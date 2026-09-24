# Ficha de diseño de roles — Fase 2 [EN CURSO]
*(borrador para Capítulo 4 — Marco Metodológico, y para anexo técnico de prompts)*

## Esquema de salida común a los tres analistas

```json
{
  "role": "technical | sentiment | fundamental",
  "signal": "bullish | neutral | bearish",
  "score": -1.0,
  "confidence": 0.0,
  "rationale": "max 50 words"
}
```

`signal` es la etiqueta categórica legible; `score` (continuo en [-1,1]) es lo que el trader agrega numéricamente entre roles; `confidence` (en [0,1]) pondera ese score en la agregación.

**Bloque de audiencia (idéntico en los tres analistas):** cada informe lo lee el trader, que decide una orden inmediatamente después del cierre del día, ejecutada a la apertura de la sesión siguiente (`execution_timing="next_open"`, confirmado en el código fuente de FINSABER-2 v2.0.1). El analista no debe recomendar una orden, solo describir el panorama.

---

## 2.1 Analista técnico [CERRADO]

- **Indicadores:** SMA20/50, RSI(14) con suavizado de Wilder, MACD(12,26,9), Bollinger(20,2), ATR(14), volumen relativo (vs. media 20d). Calculados en pandas puro (sin `ta`/`pandas-ta`), todos causales (solo datos ≤ `date`).
- **Input:** últimas 5 sesiones OHLCV + indicadores del día `date`.
- **Prototipo:** validado con `gpt-4.1-mini` sobre NVDA en 3 casos contrastados (RSI máx, RSI mín, Bollinger más estrecho), 3 repeticiones cada uno. Salidas 100% válidas, coherentes entre `signal` y `score`, y estables entre repeticiones con `temperature=0.2`.
- **Script:** `technical_prototype.py`.

---

## 2.2 Analista de sentimiento [EN DISEÑO]

- **Fuente de datos:** `news_items` filtrado por `symbol == ticker`, compartida con el rol macro/geopolítico (decisión de Fase 1.5). La diferenciación entre ambos roles está en la tarea del prompt, no en el input.
- **Ventana temporal:** últimos 3 días naturales hasta `date` inclusive. Se eligió corta porque el foco es la percepción de mercado reciente, no eventos ya descontados.
- **Control de tokens — selección por presupuesto de caracteres:** en vez de un número fijo de noticias, se seleccionan noticias (ordenadas de más a menos reciente dentro de la ventana) hasta que la suma acumulada de caracteres de `news_text` alcanza un presupuesto fijo (provisional: 6.000 caracteres). La noticia que cruza el umbral se incluye completa y no se añade ninguna más después. Pendiente de ajustar el presupuesto tras observar el volumen real por ticker en el prototipo (NVDA muy denso vs. JNJ disperso).
- **Caso sin noticias:** contemplado explícitamente en el prompt — el analista debe devolver `neutral` con `confidence` baja en vez de inventar una señal.
- **Sin filtros de contenido previos:** el analista lee cualquier noticia de la ventana tal cual llega, sin excluir ni marcar como problemáticas las que ya narran un movimiento de precio del propio día `date` ("shares jumped after..."). No es leakage: la ventana ya garantiza que solo entran noticias con fecha ≤ `date`, así que una noticia sobre el movimiento de hoy es información pública contemporánea a la decisión, igual que el cierre/volumen que usa el analista técnico. (Se había explorado un chequeo heurístico para marcar estos casos como "leakage encubierto"; se descartó por partir de esa premisa incorrecta.)
- **Terminología del prompt:** se usa "news items" en vez de "headlines", ya que `news_text` puede ser un fragmento más largo que un titular.

### Prompt (borrador)

```text
SYSTEM:
You are a sentiment analyst at an investment firm. Your report will be read by
the trader, a colleague who will use it, together with other inputs, to place
a market order for this ticker after today's close; it will be filled at the
next session's open, which is not yet known. The trader does not see the raw
data. Describe how recent news coverage is likely to be perceived by the
market over the next few trading days; do NOT recommend an order yourself.
Analyse ONLY the news items provided. Do not use outside knowledge about the
company, events after the given date, or how the market actually reacted.
If no news items are provided, say so and return a neutral signal with low
confidence. Respond with a single JSON object and nothing else.

USER:
Ticker: {ticker}
Analysis date: {date}

News items from the last 3 days (date, text):
{news_block}

Task: assess market sentiment toward {ticker} over the next few trading days,
focusing on what is most relevant for a trader deciding on an order today.
Return JSON with keys:
"role" (always "sentiment"), "signal" ("bullish"|"neutral"|"bearish"),
"score" (float in [-1,1]), "confidence" (float in [0,1]),
"rationale" (max 50 words, referencing the specific news items used).
```

### Pendiente de documentar (posibles apartados de Capítulo 6 — Prospectiva)

- **Clasificación de noticias por relevancia percibida:** como línea de trabajo futura, se identifica la posibilidad de que el sistema clasifique las noticias según su relevancia/impacto esperado, y en función de esa clasificación las tenga en cuenta durante ventanas temporales de distinta amplitud — noticias de mayor relevancia permanecerían "activas" para el analista más tiempo que noticias de impacto a corto plazo. La ventana fija de 3 días adoptada en este TFM no distingue entre ambos tipos de noticia; se documenta esta simplificación como limitación y como posible extensión futura, no implementada en el alcance de este trabajo.
- **Deduplicación de noticias:** se ha detectado que `news_items` puede contener la misma noticia repetida varias veces (sindicación de agencias/wire services). Se ha decidido **no implementar deduplicación** en el alcance de este TFM, dejándolo documentado aquí como simplificación consciente; se revisará empíricamente en el prototipo si el fenómeno es lo bastante frecuente como para sesgar el `rationale` o el `score` (por sobre-representación de una misma noticia), y en tal caso se reconsiderará.

---

## 2.3 Analista fundamental macro/geopolítico [EN DISEÑO]

- **Fuente de datos — verificado en código fuente y ficha del dataset:** `news_items` (columnas `date, symbol, cik, item_index, news_text, text_len, text_crc32, year`) no admite noticias sin ticker asociado; `symbol` es obligatorio en el filtro del loader (`FinsaberParquetDataset._read_news`) y cada fila lleva `cik`, el identificador de la SEC para una compañía concreta. Se descarta por tanto la opción de un pool de noticias macro "sin ticker" independiente del rol de sentimiento: **todo el contenido de `news_items` está indexado por compañía**.
- **Decisión de diseño:** el rol fundamental usa el mismo pool de `news_items` filtrado por ticker que el rol de sentimiento (misma ventana de 3 días, mismo presupuesto de 6.000 caracteres), diferenciándose únicamente por la tarea del prompt: se le indica explícitamente ignorar noticias específicas de la compañía sin dimensión macro/geopolítica (lanzamientos de producto, resultados sin contexto más amplio) y centrarse en tipos de interés, inflación, política monetaria, elecciones, geopolítica o condiciones sectoriales/de materias primas.
- **Enriquecimiento con `filingk`/`filingq`:** se deja fuera del prototipo inicial (el cronograma lo marca como opcional); se revisará como posible extensión si el rol resulta pobre en contenido solo con noticias.
- **Limitación documentada (Capítulo 6):** al compartir la misma fuente indexada por compañía que el rol de sentimiento, el rol fundamental puede (a) correlacionarse con sentimiento por compartir input, y (b) no capturar eventos macro puros que no mencionen directamente al ticker, ya que el dataset no ofrece un feed macro independiente de las compañías. Una fuente externa dedicada a noticias macro generales resolvería esto, pero se descarta en el alcance de este TFM.
- **Script:** `fundamental_prototype.py` (reutiliza `load_news`/`select_news`/`pick_cases` de `sentiment_prototype.py`).

### Prompt (borrador)

```text
SYSTEM:
You are a macro/geopolitical analyst at an investment firm. Your report will be read by
the trader, a colleague who will use it, together with other inputs, to place a market order
for this ticker after today's close; it will be filled at the next session's open, which is
not yet known. The trader does not see the raw data. Describe how the broader macroeconomic
and geopolitical context (interest rates, inflation, monetary policy, elections, geopolitical
events, sector-wide or commodity conditions) is likely to affect this ticker over the next few
trading days; do NOT recommend an order yourself. Ignore company-specific news that has no
macro/geopolitical dimension (e.g. product launches, earnings beats with no broader context).
Analyse ONLY the news items provided. Do not use outside knowledge about the company, events
after the given date, or how the market actually reacted. If no news items with macro/geopolitical
relevance are provided, say so and return a neutral signal with low confidence. Respond with a
single JSON object and nothing else.

USER:
Ticker: {ticker}
Analysis date: {date}

News items from the last 3 days (date, text):
{news_block}

Task: assess the macroeconomic/geopolitical outlook for {ticker} over the next few trading days,
focusing on what is most relevant for a trader deciding on an order today.
Return JSON with keys:
"role" (always "fundamental"), "signal" ("bullish"|"neutral"|"bearish"),
"score" (float in [-1,1]), "confidence" (float in [0,1]),
"rationale" (max 50 words, referencing the specific news items used).
```

## 2.4 Trader / decisor final [CERRADO]

- **Sizing de posición:** el trader decide un tamaño (`size` ∈ [0,1]) además de la acción, en vez de operar siempre al máximo (`quantity=-1`). Esta característica se mantiene **constante en las 5 configuraciones de ablación** (mismo mecanismo de traducción `size → quantity`), por lo que no introduce una variable de confusión en la ablación: lo único que varía entre configuraciones sigue siendo qué informes de analistas recibe el trader.
- **Traducción `size → quantity` — se hace en código Python, no en el LLM:**
  ```python
  if action == "buy":
      quantity = int(size * framework.cash / price)
  elif action == "sell":
      held = framework.portfolio.get(ticker, {}).get("quantity", 0)
      quantity = int(size * held)
  else:  # hold
      quantity = 0  # no se llama a framework.buy/sell
  ```
  Evita pedirle al LLM que calcule cash o nº de acciones (fuente de errores de formato). Con `size=1.0` en `buy` se recupera el comportamiento "todo el cash" (equivalente al antiguo `quantity=-1`), sin forzarlo como única opción.
- **Contexto de posición:** el prompt incluye `current position` (`no position (all cash)` | `holding shares (fully invested)`), leído de `framework.cash`/`framework.portfolio`, no inventado por el LLM. Ayuda a que el razonamiento sea coherente (no proponer "buy" con lógica de apertura si ya se lleva semanas invertido), aunque no es estrictamente necesario para la corrección del framework: `_resolve_quantity` ya resuelve órdenes redundantes a `quantity=0` de forma segura (comprobado en el código fuente).
- **Agregación con criterio propio, no mecánica:** se decide deliberadamente que el trader no promedie `score`/`confidence` con una fórmula, sino que aplique juicio (dar más peso a consensos que a un informe aislado, desconfiar de una confidence alta con rationale débil, reducir tamaño ante señales conflictivas). Justificación: si fuera agregación mecánica no haría falta un LLM en ese rol, y perdería sentido la comparación con la arquitectura de TradingAgents en la que se inspira el sistema.
- **Guardarraíl de no-conocimiento-externo:** el trader recibe la misma restricción que los tres analistas ("no uses conocimiento externo sobre la compañía ni eventos posteriores a la fecha dada"), que no estaba contemplada en el primer borrador del prompt. Sin ella, el criterio propio del trader podría convertirse en una vía de leakage a nivel de sistema, afectando por igual a las 5 configuraciones y quedando invisibilizado como "efecto de agregación".
- **Caso baseline (`cherry_pick_baseline_price_only`, sin analistas):** el bloque de informes se sustituye por el texto explícito `(no analyst reports available — decide based on current position only)`, en vez de dejarlo vacío, para documentar que es un caso contemplado a propósito.
- **Nota de consistencia de nombrado (pendiente, no toca la Fase 1 ya cerrada):** el rol se renombró de "macro" a "fundamental" (2.3), pero los `setup_name` de la ficha de Fase 1.3 siguen usando `cherry_pick_no_macro`. Queda anotado para decidir en la Fase 3 si se renombra a `cherry_pick_no_fundamental` por coherencia o se documenta la discrepancia.

### Esquema de salida

```json
{
  "action": "buy | hold | sell",
  "size": 0.0,
  "confidence": 0.0,
  "rationale": "max 60 words"
}
```

### Prompt (versión final)

```text
SYSTEM:
You are the trader at an investment firm. You receive reports from your colleague analysts and
decide the trading action for this ticker after today's close; the order will be filled at the
next session's open, which is not yet known. Not all reports are always available; base your
decision only on the ones provided below.
Use your professional judgment to weigh and reconcile the reports — you are not required to
average their scores mechanically. For example, give more weight to reports that agree with each
other than to a single outlier; treat a high-confidence report with caution if its rationale seems
weak; and prefer a smaller position size when reports conflict, even if their average score looks
strong.
Do not use outside knowledge about the company or events after the given date beyond what is
stated in the reports below. Respond with a single JSON object and nothing else.

USER:
Ticker: {ticker}
Analysis date: {date}
Current position: {position_state}

Analyst reports:
{reports_block}

Task: decide the trading action for {ticker} today, considering the analyst reports above and the
current position (e.g. "sell" only makes sense if you currently hold shares).
Return JSON with keys:
"action" ("buy"|"hold"|"sell"),
"size" (float in [0,1]: fraction of available cash to invest if buying, or fraction of the
  current position to liquidate if selling; ignored if holding),
"confidence" (float in [0,1]),
"rationale" (max 60 words, referencing which analyst reports weighed most in the decision).
```

- **Script:** `trader_prototype.py`. No llama a FINSABER-2 directamente (el trader agrega informes ya producidos); usa 7 escenarios fijos combinando 0-3 informes de analista (algunos reales, tomados de ejecuciones previas de los otros prototipos; otros sintéticos y marcados como tales para completar combinaciones no cubiertas aún por una ejecución real) y ambos estados de posición. Valida formato, dominios, y tres chequeos de coherencia: `sell` sin posición abierta, `hold` con `size != 0`, y `buy`/`sell` con `size == 0` (equivalente a un hold disfrazado).

## 2.5 Validación cruzada de formato [PENDIENTE]