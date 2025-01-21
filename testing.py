# back_testing.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

def backtest_pair_with_capital(stockA, stockB, positions, spy_series, initial_capital=10000.0):
    """
    Backtest a pairs trading strategy with a given initial capital.

    Strategy rule:
      +1 => Long A, Short B  (Invest half of capital in A, half short in B)
      -1 => Short A, Long B  (Invest half short in A, half in B)
       0 => No position (all in cash)

    We do simple daily rebalancing:
      - Each day, if pos=+1 => daily return = 0.5*A_ret - 0.5*B_ret
      - If pos=-1 => daily return = -0.5*A_ret + 0.5*B_ret
      - If pos= 0 => daily return = 0

    Also tracks how $10k would grow if invested in SPY (spy_series) on the first day.
    """
    df = pd.DataFrame({
        'A': stockA,
        'B': stockB,
        'Pos': positions
    }).dropna()

    df['A_ret'] = df['A'].pct_change().fillna(0)
    df['B_ret'] = df['B'].pct_change().fillna(0)

    capital_history = [initial_capital]
    idx = df.index

    for i in range(1, len(df)):
        prev_capital = capital_history[-1]
        pos = df['Pos'].iloc[i]

        if pos == 1:
            # +1 => 0.5*A_ret - 0.5*B_ret
            daily_r = 0.5 * df['A_ret'].iloc[i] - 0.5 * df['B_ret'].iloc[i]
        elif pos == -1:
            # -1 => -0.5*A_ret + 0.5*B_ret
            daily_r = -0.5 * df['A_ret'].iloc[i] + 0.5 * df['B_ret'].iloc[i]
        else:
            daily_r = 0.0

        current_capital = prev_capital * (1.0 + daily_r)
        capital_history.append(current_capital)

    df['Strategy_Capital'] = pd.Series(capital_history, index=idx)

    # Compare to buy-and-hold in SPY
    # Make sure spy_series aligns with df's dates
    spy_aligned = spy_series.loc[df.index].ffill()
    first_spy_price = spy_aligned.iloc[0]
    df['SPY_BuyHold'] = initial_capital * (spy_aligned / first_spy_price)

    return df[['Strategy_Capital', 'SPY_BuyHold', 'Pos']]

def plot_backtest_results(backtest_df, pair_name=""):
    """
    Plot the strategy capital vs. SPY buy & hold from the DataFrame.
    Optionally label the plot with pair_name (e.g. 'A vs. B').
    """
    plt.figure(figsize=(10, 5))
    plt.plot(backtest_df.index, backtest_df['Strategy_Capital'], label='Pairs Strategy')
    plt.plot(backtest_df.index, backtest_df['SPY_BuyHold'], label='SPY Buy & Hold')

    plt.title(f'Pairs Strategy vs. SPY Buy & Hold: {pair_name}')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.show()
