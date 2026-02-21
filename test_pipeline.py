import pandas as pd
import yfinance as yf
from pipeline import StrategyEngine, create_policy
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def test_pipeline():
    tickers = ["MSFT"]
    print(f"Downloading data for {tickers}...")
    stock_data = {}
    for t in tickers:
        df = yf.download(t, period="3mo", progress=False)
        df = df.reset_index()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        stock_data[t] = df

    print("Setting up engine...")
    policy = create_policy(tickers, [1.0], total_capital=10000, risk_tolerance="Medium", rebalance_frequency="Weekly")
    engine = StrategyEngine(policy)
    
    print("Running Day 1 Iteration...")
    res1 = engine.run(stock_data)
    print(f"Day 1 Dominant Regime: {res1.dominant_regime}")
    print(f"Day 1 Top 5 Strategies for MSFT: {res1.per_stock_details['MSFT']['candidates']}")
    print(f"Day 1 Winner: {res1.per_stock_strategies['MSFT']}")

    print("\nRunning Day 2 Iteration (to trigger Feedback Loop)...")
    res2 = engine.run(stock_data)
    print("Execution complete. Check for errors.")

if __name__ == "__main__":
    test_pipeline()
