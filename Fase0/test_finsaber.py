import os
from dotenv import load_dotenv
from finsaber import FINSABERBt, FinsaberParquetDataset
from finsaber.strategy.timing import BuyAndHoldStrategy

load_dotenv()

data = FinsaberParquetDataset(os.environ["FINSABER_DATA_PATH"])

config = {
    "data_loader": data,
    "tickers": ["AAPL"],
    "date_from": "2024-01-02",
    "date_to": "2024-01-10",
    "setup_name": "cherry_pick_demo",
    "execution_timing": "next_open",
    "slippage_perc": 0.0005,
    "liquidity_cap_pct": 0.025,
    "save_results": True,
    "silence": True,
}

results = FINSABERBt(config).run_iterative_tickers(BuyAndHoldStrategy)
print(results["AAPL"]["total_return"])