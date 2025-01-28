# analysis.py

import pandas as pd

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

def get_trades_for_stock(trade_log_df, ticker, x=5):
    """
    Filters the 'trade_log_df' for a specific ticker, prints the total number
    of trades, and returns the first 'x' trades chronologically.

    Parameter
    ----------
    trade_log_df : pd.DataFrame
        A DataFrame containing trades with columns like:
          ['Date', 'Ticker', 'Action', 'Price', ...]
    ticker : str
        The ticker symbol you want to filter. E.g. 'MSFT'
    x : int
        The number of trades to display (default 5).

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with trades for the specified ticker,
        limited to the first 'x' in chronological order.
    """
    # Filter to the chosen ticker
    filtered = trade_log_df[trade_log_df['Ticker'] == ticker].copy()

    # total trades
    total_trades = len(filtered)
    print(f"Total trades for {ticker}: {total_trades}")

    #Show only the first x trades
    if total_trades == 0:
        print(f"No trades found for {ticker}.")
        return filtered  # empty

    x = min(x, total_trades)  # in case x > total_trades
    subset = filtered.head(x)
    print(f"\nShowing the first {x} trades for {ticker}:\n")
    display(subset)

    return subset