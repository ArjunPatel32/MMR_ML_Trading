# testing.py

from computing import compute_zscore, compute_spread
from pair_trading import track_strategy, generate_positions

def backtest_pair_with_capital(
  historical_data,
  stock1_name,
  stock2_name,
  spy_series,
  window=30,
  initial_capital=10000.0,
  z_entry=1.0,
  z_exit=0.0
  ):
  """
  Runs the full backtest process for a single pair: (stock1_name, stock2_name).
  Steps:
    1) Extract each stock's series
    2) Compute dynamic spread & hedge ratio
    3) Z-score the spread
    4) Generate positions
    5) Backtest with dynamic hedge ratio
  Returns:
    A DataFrame with 'Strategy_Capital', 'SPY_BuyHold', 'Pos', 'Beta'
  """
  s1 = historical_data[stock1_name].dropna()
  s2 = historical_data[stock2_name].dropna()

  # Compute spread + dynamic hedge
  dynamic_spread, beta_series = compute_spread(s1, s2, window=window)

  # Z-score the spread
  zscore_spread = compute_zscore(dynamic_spread)

  # Generate positions
  positions = generate_positions(zscore_spread, z_entry=z_entry, z_exit=z_exit)

  # Run backtest
  result_df = track_strategy(
      s1,
      s2,
      positions,
      spy_series,
      beta_series,
      initial_capital=initial_capital
  )
  
  return result_df
