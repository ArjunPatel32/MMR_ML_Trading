# testing.py

from computing import compute_zscore, compute_spread
from pair_trading import track_strategy, generate_positions

def backtest_pair(
    historical_data,
    stock1_name,
    stock2_name,
    spy_series,
    window=30,
    initial_capital=10000.0,
    z_entry=1.0,
    z_exit=0.0,
    max_drawdown_pct=0.10,    # 10% max drawdown
    volatility_window=30,
    target_volatility=0.02    # 2% target daily volatility
):
    """
    Runs the full backtest process for a single pair: (stock1_name, stock2_name).
    """
    s1 = historical_data[stock1_name].dropna()
    s2 = historical_data[stock2_name].dropna()
    
    # Compute the spread & dynamic hedge ratio
    dynamic_spread, beta_series = compute_spread(s1, s2, window=window)
    
    # Z-score the spread
    zscore_spread = compute_zscore(dynamic_spread)
    
    # Generate positions from z-score
    positions = generate_positions(zscore_spread, z_entry=z_entry, z_exit=z_exit)
    
    # Run backtest with volatility-based risk management
    result_df = track_strategy(
        s1,
        s2,
        positions,
        spy_series,
        beta_series,
        initial_capital=initial_capital,
        max_drawdown_pct=max_drawdown_pct,
        volatility_window=volatility_window,
        target_volatility=target_volatility
    )
    
    return result_df

