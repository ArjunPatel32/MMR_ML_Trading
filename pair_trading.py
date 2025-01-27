# pair_trading.py

import pandas as pd

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
    initial_capital=10000.0,
    max_drawdown_pct=None,
    volatility_window=30,
    target_volatility=0.02
):
    """
    Optimized backtest of a pairs trading strategy with volatility-based scaling and risk management.
    """
    
    df = pd.DataFrame({
        'A': stockA,
        'B': stockB,
        'Pos': positions,   
        'Beta': beta_series
    }).dropna()
    
    # Clamp Beta
    df['Beta'] = df['Beta'].clip(-10, 10)
    
    # Calculate daily returns
    df['A_ret'] = df['A'].pct_change().fillna(0)
    df['B_ret'] = df['B'].pct_change().fillna(0)
    
    # Calculate Raw_Return
    df['Raw_Return'] = df.apply(
        lambda row: 0.5 * row['A_ret'] - 0.5 * row['Beta'] * row['B_ret'] if row['Pos'] == 1
        else (-0.5 * row['A_ret'] + 0.5 * row['Beta'] * row['B_ret']) if row['Pos'] == -1
        else 0.0,
        axis=1
    )
    
    # Calculate Rolling Volatility
    df['Rolling_Volatility'] = df['Raw_Return'].rolling(window=volatility_window).std().bfill()
    df['Rolling_Volatility'] = df['Rolling_Volatility'].replace(0, 1e-6)
    
    # Calculate Scaling Factor
    df['Scaling_Factor'] = target_volatility / df['Rolling_Volatility']
    df['Scaling_Factor'] = df['Scaling_Factor'].clip(lower=0.1, upper=3.0)
    
    # Adjusted Return
    df['Adjusted_Return'] = df['Raw_Return'] * df['Scaling_Factor']
    
    # Initialize Strategy Capital
    df['Strategy_Capital'] = initial_capital
    df['Exit_DD'] = False
    capital = initial_capital
    peak_capital = initial_capital
    
    for i in range(1, len(df)):
        adjusted_r = df['Adjusted_Return'].iloc[i]
        capital *= (1.0 + adjusted_r)
        
        # Max Drawdown Check
        if max_drawdown_pct is not None:
            peak_capital = max(peak_capital, capital)
            drawdown = (peak_capital - capital) / peak_capital
            if drawdown > max_drawdown_pct:
                capital = peak_capital
                df.at[df.index[i], 'Pos'] = 0  # Exit position
                df.at[df.index[i], 'Exit_DD'] = True
        
        df.at[df.index[i], 'Strategy_Capital'] = capital
    
    # SPY Buy and Hold
    spy_aligned = spy_series.loc[df.index].ffill()
    first_spy_price = spy_aligned.iloc[0]
    df['SPY_BuyHold'] = initial_capital * (spy_aligned / first_spy_price)
    
    return df[['A', 'B', 'Pos', 'Strategy_Capital', 'SPY_BuyHold', 'Beta', 'Exit_DD']]





