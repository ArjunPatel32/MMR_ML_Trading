# data_collection.py

import yfinance as yf
import pandas as pd

def get_historical_data(tickers, start="2023-12-31", end="2024-12-31", interval='1d'):
    """
    This function returns a pd.DataFrame with the closing prices of the given tickers.
    """
    data = pd.DataFrame()
    for symbol in tickers:
        df = yf.download(symbol, start=start, end=end, interval=interval)
        data[symbol] = df['Close']
    return data

def detect_stock_pairs(historical_data, corr_threshold=0.8, value_diff_threshold=0.01):
    """
    Identifies highly correlated stock pairs from historical data based on a correlation threshold
    and ensures their values are similar enough to intersect at some points.
    """
    corr_matrix = historical_data.corr()
    high_corr_pairs = []
    
    for stock1 in corr_matrix.columns:
        for stock2 in corr_matrix.columns:
            if stock1 != stock2 and corr_matrix.loc[stock1, stock2] > corr_threshold:
                # Check if the stocks have intersecting or close values
                s1_vals = historical_data[stock1]
                s2_vals = historical_data[stock2]
                close_values = (
                    abs(s1_vals - s2_vals) / ((s1_vals + s2_vals) / 2)
                ) < value_diff_threshold
                if close_values.any():
                    high_corr_pairs.append((stock1, stock2, corr_matrix.loc[stock1, stock2]))
    
    # Remove duplicates and sort
    unique_high_corr_pairs = list(
        set(tuple(sorted(pair[:2])) + (pair[2],) for pair in high_corr_pairs)
    )
    high_corr_pairs_df = pd.DataFrame(
        unique_high_corr_pairs, columns=["Stock 1", "Stock 2", "Correlation"]
    ).sort_values(by="Correlation", ascending=False)
    
    return high_corr_pairs_df

