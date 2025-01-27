# pair_trading.py

import statsmodels.tsa.stattools as ts
import pandas as pd

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

def generate_signals(zscore, z_entry=1.0, z_exit=0.0):
    """
    Generate raw signals (+1, 0, -1) based on a z-score threshold:
      +1 => zscore < -z_entry (long the ratio)
      -1 => zscore >  z_entry (short the ratio)
       0 => otherwise
    """
    signal = pd.Series(0.0, index=zscore.index)
    signal[zscore < -z_entry] = 1.0
    signal[zscore >  z_entry] = -1.0
    return signal

def generate_positions(zscore, z_entry=1.0, z_exit=0.0):
    """
    Convert signals into 'positions' (with a simple stateful exit rule):
      - If we have no position and see +1 or -1, we open that position.
      - Once in a position, we exit when zscore crosses back through 0.
    """
    raw_signals = generate_signals(zscore, z_entry=z_entry, z_exit=z_exit)
    positions = []
    position_state = 0

    for i in range(len(raw_signals)):
        if position_state == 0:
            # open position if signal is ±1
            if raw_signals.iloc[i] == 1.0:
                position_state = 1
            elif raw_signals.iloc[i] == -1.0:
                position_state = -1
        else:
            # if we're long and zscore > 0 => exit
            if position_state == 1 and zscore.iloc[i] > 0:
                position_state = 0
            # if we're short and zscore < 0 => exit
            elif position_state == -1 and zscore.iloc[i] < 0:
                position_state = 0

        positions.append(position_state)

    return pd.Series(positions, index=zscore.index)

def track_strategy(
    stockA, 
    stockB, 
    positions, 
    spy_series, 
    beta_series, 
    initial_capital=10000.0
):
    """
    Backtest a pairs trading strategy with a given initial capital.

    We do simple daily rebalancing with a dynamic hedge ratio (beta_series) 
    and also track SPY buy-and-hold.
    """
    df = pd.DataFrame({
        'A': stockA,
        'B': stockB,
        'Pos': positions,    # +1, -1, or 0
        'Beta': beta_series
    }).dropna()

    # Compute daily returns for each stock
    df['A_ret'] = df['A'].pct_change().fillna(0)
    df['B_ret'] = df['B'].pct_change().fillna(0)

    # Track strategy capital
    capital_history = [initial_capital]
    idx = df.index

    for i in range(1, len(df)):
        prev_capital = capital_history[-1]
        pos = df['Pos'].iloc[i]
        beta_today = df['Beta'].iloc[i]

        if pos == 1:
            daily_r = 0.5 * df['A_ret'].iloc[i] - 0.5 * beta_today * df['B_ret'].iloc[i]
        elif pos == -1:
            daily_r = -0.5 * df['A_ret'].iloc[i] + 0.5 * beta_today * df['B_ret'].iloc[i]
        else:
            daily_r = 0.0

        current_capital = prev_capital * (1.0 + daily_r)
        capital_history.append(current_capital)

    df['Strategy_Capital'] = pd.Series(capital_history, index=idx)

    # Track SPY Buy & Hold
    spy_aligned = spy_series.loc[df.index].ffill()
    first_spy_price = spy_aligned.iloc[0]
    df['SPY_BuyHold'] = initial_capital * (spy_aligned / first_spy_price)

    # Return DataFrame including A, B, and signals for plotting
    return df[['A','B','Pos','Strategy_Capital','SPY_BuyHold','Beta']]


