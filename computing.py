# computing.py

import numpy as np

def compute_momentum(data, window=126):
    """
    Compute momentum as the percentage change over a given window.
    """
    return data.pct_change(periods=window)

def compute_mean_reversion(data, window=20):
    """
    Compute mean reversion signals using z-scores.
    """
    # Calculate rolling mean and standard deviation
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()

    # Calculate z-score
    z_score = (data - rolling_mean) / rolling_std
    return z_score

def compute_signal_returns(data, signal_df):
    """
    data: price DataFrame (each column is a ticker)
    signal_df: signal DataFrame (matching index/tickers), with values in {-1,0,1}.

    Returns a Series of "portfolio daily return" for each day,
    assuming we equally weight all tickers that have a non-zero signal.
    """
    # Daily returns
    daily_returns = data.pct_change().shift(-1)  # shift(-1) so day t signal sees day t+1 return

    # If signal is +1 and daily_return is r, that's +r. If -1, it's -r. If 0, it's 0.
    # Average across "active" tickers
    combined = (signal_df * daily_returns)

    # equal-weight average of all active signals:
    combined['sum'] = combined.sum(axis=1)
    combined['count'] = (signal_df != 0).sum(axis=1).replace(0, np.nan)
    combined['strategy_return'] = combined['sum'] / combined['count']

    # That gives daily strategy return for each day.
    # per-day return:
    return combined['strategy_return'].fillna(0.0)