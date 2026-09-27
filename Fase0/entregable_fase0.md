# Fase 0 — Validación de viabilidad
## Entregable

---

## 0.1 — Azure OpenAI (vía Azure AI Foundry)

- [x] Acceso confirmado: Sí
- Recurso: Azure AI Foundry (no el recurso "Azure OpenAI" clásico)
  - Nota: gpt-4o-mini está deprecated → se descartó
- Modelo desplegado: **gpt-4.1-mini** (tipo "Standard"/Global Standard, sin servidor)
- SDK y API:
  - Librería: `openai.OpenAI` (NO `AzureOpenAI`)
  - Endpoint: `https://foundry-tfm-trading.services.ai.azure.com/openai/v1`
  - Método: `client.responses.create()` (Responses API), NO `chat.completions.create()`
  - Extracción de texto: `response.output_text`
- Cuota confirmada:
  - TPM (tokens/minuto): 100.000
  - RPM (requests/minuto): 100
- Configuración de red: "Todas las redes, incluso Internet" (adecuado para desarrollo individual)
- Fecha de caducidad del crédito: suficientemente tarde como para finalizar el TFM
- Coste estimado de uso: pendiente de calcular en Fase 1/3 según volumen de llamadas proyectado

---

## 0.2 — Gemini API (free tier)

- [x] Acceso confirmado: Sí
- API: **Interactions API** (sustituye a la antigua `generate_content`)
  - `gemini-2.0-flash` y `gemini-3.6-flash` en tier estándar → cuota insuficiente (RPD=20) o deprecated
- Modelo final elegido: **gemini-3.5-flash-lite**
  - RPM: 15 | TPM: 250.000 | RPD: 500
- SDK: `google-genai`
  - Método: `client.interactions.create()`
  - Extracción de texto: `interaction.output_text`
- **Rol asignado en el diseño**: fallback puntual ante fallos de Azure (rate limit, timeout, error de formato) + validación cruzada de formato de prompts (punto 2.5 del cronograma)
  - **No se usará como motor principal de ningún rol** — decisión tomada por fragilidad del tier gratuito (cambios de modelo/API observados durante esta misma fase)

---

## 0.3 — FINSABER-2
- [x] Instalación correcta: Sí
- [x] Quick Start ejecutado con éxito: Sí (total_return = 0.00443)
- [x] Persistencia de resultados confirmada y entendida:
  - Se activa siempre que save_results=True (valor por defecto), sin depender del nombre del setup_name
  - Ruta: log_base_dir/setup_name/strategy_name/date_from_date_to/ticker/
  - Artefactos generados: equity_curve.csv, metrics.json, orders.csv (+ trades.csv,
    rejected_orders.csv, llm_costs.csv cuando aplique)
  - log_base_dir por defecto: dentro de site-packages del venv (no ideal)

---

## 0.4 — Interfaz de estrategia
- [x] Validado con código fuente real (no solo documentación)
- Clase base: BaseStrategyIso (finsaber.strategy.timing_llm)
  - Método a implementar: on_data(self, date, today_data, framework)
  - today_data: dict del día concreto (via data_loader.get_data_by_date(date))
- Framework: FINSABERFrameworkHelper
  - framework.buy(date, ticker, price, quantity) / .sell(...)
  - quantity=-1 → usa todo el cash disponible / vende toda la posición (cuidado con multi-ticker)
  - Gestiona automáticamente: comisión, slippage, liquidity cap, rechazo de órdenes
- LLM cost accounting: NO automático
  - Debe invocarse explícitamente: add_openai_cost_from_response(response) en el wrapper call_llm()
  - ⚠️ Bug/limitación detectada: OPENAI_PRICE no incluye "gpt-4.1-mini" → coste se registraría como 0.0
  - Acción para Fase 3.2: parchear OPENAI_PRICE con precio real de gpt-4.1-mini antes de usar el monitor
  - Gemini no está soportado por este monitor (provider="openai" únicamente) — coste manual si se usa
