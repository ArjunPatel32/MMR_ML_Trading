# analysis.py

import pandas as pd
import matplotlib.pyplot as plt

def count_strategy_choices(clf, features):
    """
    Count how many times the classifier picks momentum (1)
    vs. mean reversion (0), based on the 'features' DataFrame.
    """

    # Align features and drop NAs
    X = features.dropna()
    # Get predictions (0 = MR, 1 = Momentum)
    predictions = clf.predict(X)

    # Count how many days for each
    import numpy as np
    n_momentum = np.sum(predictions == 1)
    n_meanrev = np.sum(predictions == 0)

    print(f"Momentum chosen:       {n_momentum} days")
    print(f"Mean Reversion chosen: {n_meanrev} days")

    return n_momentum, n_meanrev

def build_trade_log(final_signals, price_data):
    """
    Creates a simplified "trade log" showing each time a position is opened or closed.
    
    final_signals: DataFrame of signals in {-1, 0, +1} (index=dates, columns=tickers)
    price_data:    DataFrame of prices for each ticker (same shape or superset of final_signals)
    
    Returns a DataFrame with columns:
      - Date
      - Ticker
      - Action ("OPEN_LONG", "CLOSE_LONG", "OPEN_SHORT", "CLOSE_SHORT")
      - Price
    """
    # Ensure the price_data covers the same date range
    price_data = price_data.loc[final_signals.index, final_signals.columns].ffill()

    # Shift signals by 1 day to compare changes
    prev_signals = final_signals.shift(1).fillna(0)

    logs = []
    for date in final_signals.index:
        for ticker in final_signals.columns:
            current_signal = final_signals.at[date, ticker]
            previous_signal = prev_signals.at[date, ticker]

            # If no change, skip
            if current_signal == previous_signal:
                continue

            # Check transitions
            # e.g. 0 -> +1 means "OPEN_LONG"
            if previous_signal == 0 and current_signal == +1:
                logs.append({
                    'Date': date,
                    'Ticker': ticker,
                    'Action': 'OPEN_LONG',
                    'Price': price_data.at[date, ticker]
                })
            elif previous_signal == +1 and current_signal == 0:
                logs.append({
                    'Date': date,
                    'Ticker': ticker,
                    'Action': 'CLOSE_LONG',
                    'Price': price_data.at[date, ticker]
                })
            elif previous_signal == 0 and current_signal == -1:
                logs.append({
                    'Date': date,
                    'Ticker': ticker,
                    'Action': 'OPEN_SHORT',
                    'Price': price_data.at[date, ticker]
                })
            elif previous_signal == -1 and current_signal == 0:
                logs.append({
                    'Date': date,
                    'Ticker': ticker,
                    'Action': 'CLOSE_SHORT',
                    'Price': price_data.at[date, ticker]
                })
            # flips from +1 to -1 directly
            elif previous_signal == +1 and current_signal == -1:
                # close the long and open the short
                logs.append({
                    'Date': date,
                    'Ticker': ticker,
                    'Action': 'CLOSE_LONG',
                    'Price': price_data.at[date, ticker]
                })
                logs.append({
                    'Date': date,
                    'Ticker': ticker,
                    'Action': 'OPEN_SHORT',
                    'Price': price_data.at[date, ticker]
                })
            elif previous_signal == -1 and current_signal == +1:
                logs.append({
                    'Date': date,
                    'Ticker': ticker,
                    'Action': 'CLOSE_SHORT',
                    'Price': price_data.at[date, ticker]
                })
                logs.append({
                    'Date': date,
                    'Ticker': ticker,
                    'Action': 'OPEN_LONG',
                    'Price': price_data.at[date, ticker]
                })

    trade_log_df = pd.DataFrame(logs)
    trade_log_df.sort_values(by='Date', inplace=True)
    trade_log_df.reset_index(drop=True, inplace=True)
    return trade_log_df

def get_trades_for_stock(
    trade_log_df,
    final_signals,
    price_data,
    ticker,
    x=5,
    initial_capital=10000.0
):
    """
    1. Filters 'trade_log_df' for a specific 'ticker', prints the total number
       of trades, and displays the first 'x' trades in chronological order.
    2. Plots an equity curve showing how the strategy would have performed if
       we ONLY traded this single ticker with the signals in 'final_signals'.
    3. On the same plot, also shows a "buy & hold" line for that ticker
       for direct comparison.

    Parameters
    ----------
    trade_log_df : pd.DataFrame
        A DataFrame containing trades with columns like:
          ['Date', 'Ticker', 'Action', 'Price', ...]
    final_signals : pd.DataFrame
        Signals in {-1, 0, +1}, indexed by date, columns = tickers.
    price_data : pd.DataFrame
        Historical price data, indexed by date, columns = tickers.
    ticker : str
        The ticker symbol to filter. E.g. 'MSFT'.
    x : int
        How many trades to display (the first x). Default is 5.
    initial_capital : float
        Starting capital for the single-ticker equity curve.
    """


    # Filter trade log for this ticker
    filtered = trade_log_df[trade_log_df['Ticker'] == ticker].copy()
    total_trades = len(filtered)

    print(f"Total trades for {ticker}: {total_trades}\n")

    # Show first 'x' trades
    if total_trades == 0:
        print(f"No trades found for {ticker}.")
    else:
        x = min(x, total_trades)
        subset = filtered.head(x)
        print(f"Showing the first {x} trades for {ticker}:\n")
        print(subset)


    # Build the single-ticker strategy equity curve
    # Reindex signals & price data to ensure alignment
    ticker_signals = final_signals[ticker].reindex(price_data.index).fillna(0)

    # Daily returns for the ticker
    daily_returns = price_data[ticker].pct_change().fillna(0)

    # Strategy daily return = signal * daily return
    # Shift signals by 1 day if your logic is "trade next day after signal"
    strategy_daily_return = ticker_signals.shift(1).fillna(0) * daily_returns

    # Cumulative product of returns => equity curve
    equity_curve = (1.0 + strategy_daily_return).cumprod() * initial_capital


    # Build the buy & hold equity curve
    # Assume buy & hold from the very first available price:
    price_series = price_data[ticker].dropna()
    if price_series.empty:
        print(f"\nNo valid price data found for {ticker}, cannot plot buy & hold.")
        # Just plot strategy then return
        plt.figure(figsize=(10, 6))
        plt.plot(equity_curve.index, equity_curve, label=f"{ticker} Strategy")
        plt.title(f"Single-Ticker Strategy vs. Buy & Hold: {ticker}")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value ($)")
        plt.legend()
        plt.show()
        return filtered, equity_curve

    start_price = price_series.iloc[0]
    buy_hold_equity = (price_series / start_price) * initial_capital

    # Align buy_hold_equity to the strategy's index
    # (Forward-fill so both lines line up on the same date range)
    buy_hold_equity = buy_hold_equity.reindex(equity_curve.index, method='ffill')

    # Plot both curves
    plt.figure(figsize=(10, 6))
    plt.plot(equity_curve.index, equity_curve, label=f"{ticker} Strategy")
    plt.plot(buy_hold_equity.index, buy_hold_equity, label=f"{ticker} Buy & Hold", linestyle='--')
    plt.title(f"Single-Ticker Strategy vs. Buy & Hold: {ticker}")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.show()

    return filtered, equity_curve, buy_hold_equity