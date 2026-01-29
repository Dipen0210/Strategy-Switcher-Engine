import pandas as pd

def generate_signals(
    position_sizes: pd.DataFrame,
    current_portfolio: dict = None
) -> pd.DataFrame:
    """
    Generate actionable buy/sell signals based on target position sizes.
    
    Args:
        position_sizes: DataFrame with target allocations (USD)
        current_portfolio: Optional dict of current holdings {ticker: value}
        
    Returns:
        DataFrame with 'Signal' (Buy/Sell/Hold) and 'Trade_Value'
    """
    # Simple pass-through for now, can be expanded
    signals = position_sizes.copy()
    signals["Signal"] = "REBALANCE"
    return signals
