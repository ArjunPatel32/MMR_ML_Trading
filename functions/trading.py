# trading.py

import numpy as np
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
    daily_returns = price_data.pct_change(fill_method='pad')

    # rolling volatility (mean across all tickers)
    # 20-day rolling std, then average across tickers
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
    # Reindex to match signals without forward filling
    price_data = price_data.reindex(final_signals.index)
    
    # Compute daily returns using available data only
    daily_returns = price_data.pct_change().shift(-1)
    daily_returns = daily_returns.fillna(0)
    
    # Ensure signals are only used when the corresponding price is available
    final_signals_aligned = final_signals.where(price_data.notna(), 0)
    
    available_counts = price_data.notna().sum(axis=1).replace(0, np.nan)
    strategy_daily_return = (final_signals_aligned * daily_returns).sum(axis=1) / available_counts
    strategy_daily_return = strategy_daily_return.fillna(0)
    
    strategy_cumulative = (1 + strategy_daily_return).cumprod() * initial_capital
    
    # SPY Buy & Hold calculation remains as before
    spy_aligned = spy_series.reindex(strategy_cumulative.index).ffill()
    first_spy_price = spy_aligned.iloc[0]
    spy_cumulative = (spy_aligned / first_spy_price) * initial_capital
    
    # For Equal Weight Buy & Hold, calculate returns only where data exists
    daily_returns_eq = price_data.pct_change().fillna(0)
    available_counts_eq = price_data.notna().sum(axis=1).replace(0, np.nan)
    equal_weight_daily_return = daily_returns_eq.sum(axis=1) / available_counts_eq
    equal_weight_daily_return = equal_weight_daily_return.fillna(0)
    equal_weight_buy_hold = (1 + equal_weight_daily_return).cumprod() * initial_capital
    
    result_df = pd.DataFrame({
        'Strategy': strategy_cumulative,
        'SPY_BuyHold': spy_cumulative,
        'EqualWeight_BuyHold': equal_weight_buy_hold
    }, index=strategy_cumulative.index)
    
    return result_df
