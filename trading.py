# trading.py

import pandas as pd

def build_strategy_choice_label(momentum_returns, meanrev_returns):
    """
    Returns a Series of {0,1} for each date:
    1 = momentum was better
    0 = mean-reversion was better or same
    """
    label = (momentum_returns > meanrev_returns).astype(int)
    return label

def build_feature_matrix(price_data, momentum_df, zscore_df, vol_window=20):
    """
    Creates daily features for the strategy chooser:
      1. Average momentum across tickers
      2. Average zscore across tickers
      3. Rolling volatility (using daily returns)
      4. Rolling average returns (20-day)

    price_data: DataFrame of close prices for all tickers
    momentum_df: DataFrame of momentum values for all tickers
    zscore_df: DataFrame of z-scores for all tickers
    vol_window: int, rolling window for volatility
    """
    # daily returns
    daily_returns = price_data.pct_change()

    # rolling volatility (mean across all tickers)
    #20-day rolling std, then average across tickers
    rolling_volatility = daily_returns.rolling(vol_window).std().mean(axis=1)

    # rolling average of daily returns (20-day), across all tickers
    rolling_mean_returns = daily_returns.rolling(vol_window).mean().mean(axis=1)

    # average momentum & zscore across all tickers
    daily_momentum_mean = momentum_df.mean(axis=1)
    daily_zscore_mean   = zscore_df.mean(axis=1)

    # Build the feature DataFrame
    features = pd.DataFrame({
        'momentum_mean': daily_momentum_mean,
        'zscore_mean': daily_zscore_mean,
        'rolling_vol': rolling_volatility,
        'rolling_mean_ret': rolling_mean_returns
    })

    return features

def track_strategy_chosen_signals(price_data, final_signals, spy_series, initial_capital=10000.0):
    """
    A simplified version that just invests (equally) in whichever signals are 'on' each day.
    """
    # daily returns
    daily_returns = price_data.pct_change().shift(-1).fillna(0)

    # strategy returns
    combined = (final_signals * daily_returns).mean(axis=1)  # average across all tickers signaled
    strategy_cumulative = (1 + combined).cumprod() * initial_capital

    # SPY buy-hold
    spy_aligned = spy_series.loc[strategy_cumulative.index].ffill()
    first_spy_price = spy_aligned.iloc[0]
    spy_cumulative = (spy_aligned / first_spy_price) * initial_capital

    result_df = pd.DataFrame({
        'Strategy': strategy_cumulative,
        'SPY_BuyHold': spy_cumulative
    }, index=strategy_cumulative.index)
    return result_df
