# computing.py

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

def compute_spread(series1, series2, window=30):
    """
    Computes a rolling hedge ratio (beta) for series1 and series2, 
    and returns a spread = series1 - beta*series2.
    
    :param series1: pd.Series of first stock's prices
    :param series2: pd.Series of second stock's prices
    :param window: rolling window size (in days)
    :return: pd.Series of the dynamic spread, pd.Series of hedge ratio
    """
    # Calculate rolling covariance and variance
    rolling_cov = series1.rolling(window).cov(series2)
    rolling_var = series2.rolling(window).var()

    # Hedge ratio (beta) = Cov(A,B) / Var(B)
    beta_series = rolling_cov / rolling_var
    
    # Spread = A - (beta * B)
    dynamic_spread = series1 - (beta_series * series2)

    return dynamic_spread, beta_series