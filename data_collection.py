# data_collection.py

import pandas as pd
import numpy as np
import yfinance as yf

def get_historical_data(tickers, start="2015-01-01", end="2025-01-01", interval='1d'):
    """
    This function returns a pd.DataFrame with the closing prices of the given tickers.
    """
    data = pd.DataFrame()
    for symbol in tickers:
        df = yf.download(symbol, start=start, end=end, interval=interval)
        data[symbol] = df['Close']
    return data

def get_sp500_sample(n=10, seed=None):
    """
    Scrapes the S&P 500 companies from Wikipedia and returns a random sample of 'n' tickers.

    Parameters:
        n (int): Number of tickers to sample.
        seed (int, optional): Seed for reproducibility. Default is None.

    Returns:
        list: A list of randomly selected ticker symbols from the S&P 500.
    """
    # URL with S&P 500 company information
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    
    # Read the table from Wikipedia
    tables = pd.read_html(url, header=0)
    sp500_table = tables[0]
    
    # Get the list of tickers. Some tickers (e.g., BRK.B) might have a period which yfinance expects as a dash.
    tickers = sp500_table['Symbol'].tolist()
    tickers = [ticker.replace('.', '-') for ticker in tickers]
    
    # Set seed for reproducibility, if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Randomly sample n tickers without replacement
    sample_tickers = np.random.choice(tickers, size=n, replace=False)
    
    return list(sample_tickers)