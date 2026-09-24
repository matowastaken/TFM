import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()
DATA_PATH = os.environ["FINSABER_DATA_PATH"]

# 1. Cargar precios de 2024 y 2025 (si existe) para ver símbolos disponibles post-cutoff
price_2024 = pd.read_parquet(os.path.join(DATA_PATH, "price_daily", "year=2024", "part-000.parquet"))
print(f"Símbolos únicos en 2024: {price_2024['symbol'].nunique()}")
print(f"Rango de fechas 2024: {price_2024['date'].min()} a {price_2024['date'].max()}")

# ¿Existe 2025?
path_2025 = os.path.join(DATA_PATH, "price_daily", "year=2025", "part-000.parquet")
if os.path.exists(path_2025):
    price_2025 = pd.read_parquet(path_2025)
    print(f"\nSímbolos únicos en 2025: {price_2025['symbol'].nunique()}")
    print(f"Rango de fechas 2025: {price_2025['date'].min()} a {price_2025['date'].max()}")
else:
    print("\nNo existe fichero para 2025")

# 2. Candidatos de perfil — verificar presencia y continuidad en 2024 (+2025 si existe)
candidatos = ["NVDA", "TSLA", "AAPL", "KO", "PG", "JNJ", "XOM", "CVX", "JPM", "GS"]
for t in candidatos:
    presente_2024 = t in price_2024['symbol'].values
    n_dias = (price_2024['symbol'] == t).sum() if presente_2024 else 0
    print(f"{t}: presente en 2024 = {presente_2024}, nº días cotizados = {n_dias}")

# 3. Cargar news_items de 2024 (y 2025 si existe) y ver símbolos con cobertura
news_2024 = pd.read_parquet(os.path.join(DATA_PATH, "news_items", "year=2024", "part-000.parquet"))
print(f"\nColumnas news_items: {news_2024.columns.tolist()}")
print(f"Símbolos únicos con noticias en 2024: {news_2024['symbol'].nunique() if 'symbol' in news_2024.columns else 'columna symbol no encontrada, revisar nombre'}")

path_news_2025 = os.path.join(DATA_PATH, "news_items", "year=2025", "part-000.parquet")
if os.path.exists(path_news_2025):
    news_2025 = pd.read_parquet(path_news_2025)
    print(f"Rango fechas news 2025: {news_2025['date'].min() if 'date' in news_2025.columns else 'revisar columna fecha'}")

# 4. Cruce: candidatos con cobertura de noticias en 2024
if 'symbol' in news_2024.columns:
    for t in candidatos:
        n_noticias = (news_2024['symbol'] == t).sum()
        print(f"{t}: nº noticias 2024 = {n_noticias}")