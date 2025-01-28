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