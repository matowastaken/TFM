# Ficha de diseño experimental — Fase 1 [CERRADA]
*(borrador para Capítulo 4 — Marco Metodológico)*

## 1.1 Objetivo de investigación

**Pregunta de investigación:** ¿Qué contribución marginal aporta cada rol especializado (técnico, sentimiento, macro/geopolítico) al performance de un sistema multiagente de trading LLM, tomando como referencia la configuración mínima del propio sistema (trader-only)?

**Objetivo general:** Evaluar la contribución marginal de cada rol especializado en un sistema multiagente LLM de trading mediante un estudio de ablación controlado.

**Referencias de comparación:**
- *Interna (ablación):* trader-only, como punto cero que comparte arquitectura con el resto de configuraciones.
- *Externa (benchmark de mercado):* buy-and-hold, usado solo en el análisis (Fase 5), no en el diseño de ablación.

**Performance:** concepto amplio (retorno bruto + métricas ajustadas a riesgo), sin priorizar una sobre otra a nivel de pregunta.

**Fuera de alcance:** la Coordination Primacy Hypothesis (Nguyen & Pham) no se asume ni se compromete a validación; se reserva como línea especulativa para Marco Teórico/Prospectiva.

## 1.2 Tickers y periodo de test

**Tickers (4, uno por perfil):**

| Ticker | Perfil | Justificación |
|---|---|---|
| NVDA | Tech alta volatilidad | Máxima cobertura mediática (20.211 noticias/2024) |
| JNJ | Defensivo | Cobertura homogénea (1.708 noticias/2024), sector salud |
| XOM | Macro/geopolítico (energía) | Sensible a geopolítica/commodities (2.210 noticias/2024) |
| JPM | Macro/tipos de interés (financiero) | Sensible a política monetaria (1.516 noticias/2024) |

**Periodo de test:** agosto 2024 - enero 2025 (6 meses). Post-cutoff de `gpt-4.1-mini` (jun-2024) con margen de seguridad; dentro de la cobertura de `news_items` (2009-2025); incluye elecciones EEUU (nov-2024) como evento macro relevante.

**Cláusula de flexibilidad:** si la carga de ejecución/QA (Fase 4) resulta excesiva, se reducirá a una ventana corta (2-3 meses) sin alterar el resto del diseño.

**Nota sobre leakage:** se documentará como limitación (Capítulo 6) que un posible leakage residual no necesariamente afecta por igual a las 5 configuraciones — p. ej., si el modelo recuerda mejor eventos macro que patrones técnicos diarios, podría inflar artificialmente el rol macro frente al técnico.

## 1.3 Configuraciones de ablación

| # | Configuración | `setup_name` | Roles activos |
|---|---|---|---|
| 1 | Sistema completo | `cherry_pick_full_system` | Técnico + Sentimiento + Macro + Trader |
| 2 | Sin técnico | `cherry_pick_no_technical` | Sentimiento + Macro + Trader |
| 3 | Sin sentimiento | `cherry_pick_no_sentiment` | Técnico + Macro + Trader |
| 4 | Sin macro | `cherry_pick_no_macro` | Técnico + Sentimiento + Trader |
| 5 | Baseline (trader-only) | `cherry_pick_baseline_price_only` | Solo Trader |

**Gestión de posición:** backtest independiente por ticker, capital inicial propio, sin pool de cash compartido — evita confundir interacción de cartera con la señal de ablación por rol. **20 backtests totales** (5 configuraciones × 4 tickers).

## 1.4 Métricas de evaluación

| Categoría | Métrica |
|---|---|
| Retorno bruto | Retorno total, retorno anualizado |
| Ajustado a riesgo | Sharpe ratio, Calmar ratio |
| Riesgo/drawdown | Max Drawdown |
| Actividad | Número de operaciones |
| Coste | Coste total de LLM (`llm_cost_monitor`, parcheado en Fase 0.4) |

**Agregación:** promedio entre los 4 tickers por configuración (comparación global) + desglose por ticker (detección de dependencia de perfil de activo).

**Delta frente a `cherry_pick_full_system`:** diferencia porcentual para métricas de retorno; diferencia absoluta para ratios, drawdown y nº de operaciones.

**Benchmark externo:** buy-and-hold por ticker, contexto en Fase 5, excluido del cálculo de deltas.

## 1.5 Fuente de noticias

**Fuente:** `news_items` de FINSABER-V2-Data, filtrado por `symbol` == ticker, compartida entre los roles de sentimiento y macro/geopolítico.

**Diferenciación por tarea (no por fuente):**
- **Sentimiento:** repercusión en la percepción de mercado (alcista/bajista, plazo).
- **Macro/geopolítico:** impacto fundamental sobre precio y contexto económico/geopolítico más amplio.

**Limitación documentada (Capítulo 6):** al compartir input, los outputs de sentimiento y macro pueden estar correlacionados pese a tareas distintas, dificultando distinguir "rol poco útil" de "roles redundantes entre sí".

**Fuente externa (NewsAPI):** descartada — cobertura ya suficiente (1.000-20.000 noticias/2024 por ticker). e inviabilidad técnica del free tier: el plan gratuito ("Developer") de NewsAPI.org está restringido a desarrollo/pruebas, con búsqueda histórica limitada a los últimos 30 días desde el momento de la consulta.