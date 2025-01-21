# pair_trading.py

import pandas as pd
import statsmodels.tsa.stattools as ts

def engle_granger_test(series1, series2):
    """
    Perform Engle-Granger cointegration test on two series.
    Returns p-value, test statistic, and critical values.
    """
    result = ts.coint(series1, series2)
    return {
        't_stat': result[0],
        'p_value': result[1],
        'crit_values': result[2]
    }

def get_ratio(series1, series2):
    """
    Return the price ratio of series1 to series2.
    """
    return series1 / series2

def compute_moving_averages(ratio, short_window=5, long_window=20):
    """
    Compute short and long moving averages of the ratio, and rolling std.
    Returns ratios_mavg_short, ratios_mavg_long, std_long
    """
    mavg_short = ratio.rolling(window=short_window).mean()
    mavg_long = ratio.rolling(window=long_window).mean()
    std_long = ratio.rolling(window=long_window).std()
    return mavg_short, mavg_long, std_long

def compute_zscore(ratio, short_window=5, long_window=20):
    """
    Compute z-score of short-window ratio vs. long-window ratio.
    """
    mavg_short, mavg_long, std_long = compute_moving_averages(ratio, short_window, long_window)
    zscore = (mavg_short - mavg_long) / std_long
    return zscore
