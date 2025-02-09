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

def get_current_sp500_ticker_sample(n=10, seed=None):
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

def get_sp500_data_in_date_range(start_date, end_date, data_csv="SPY_500_data.csv", composition_csv="SPY_500_historical_stocks.csv"):
    """
    Gets historical closing price data for the S&P 500 companies based on the composition
    at the given start_date. If the data CSV already exists and covers the requested date range,
    it simply loads and returns the sliced data. Otherwise, it downloads the data from yfinance,
    saves it to the CSV, and returns the data.

    Parameters:
        start_date (str): The start date (e.g., "2015-01-01").
        end_date (str): The end date (e.g., "2025-01-01").
        data_csv (str): Filename for saving/loading the data.
        composition_csv (str): Filename for the S&P 500 composition CSV.

    Returns:
        price_data_df (pd.DataFrame): DataFrame of historical closing prices for all tickers.
    """
    # Convert dates to Timestamps
    start_ts = pd.to_datetime(start_date)
    end_ts   = pd.to_datetime(end_date)
    
    saved_data = pd.read_csv(data_csv, index_col=0, parse_dates=True)
    # Check if the saved data covers the requested date range.
    if not saved_data.empty:
        data_start = saved_data.index.min()
        data_end   = saved_data.index.max()
        if data_start <= start_ts and data_end >= end_ts:
            # Return the slice corresponding to the requested range.
            print("Data already downloaded. Returning existing data slice.")
            return saved_data.loc[start_date:end_date]
        else:
            print("Existing data does not cover the requested date range. Redownloading data.")
    else:
        print("Saved data is empty. Redownloading data.")
    

    # Downloading Data
    # Read the S&P 500 composition CSV.
    composition_df = pd.read_csv(composition_csv, parse_dates=["date"])
    composition_df.sort_values("date", inplace=True)
    
    # Use the start_date as the reference for the composition.
    composition_at_start = composition_df[composition_df["date"] <= start_ts].iloc[-1]
    tickers_str = composition_at_start["tickers"]
    tickers = [t.strip() for t in tickers_str.split(",") if t.strip()]
    
    # Ensure SPY is included for benchmarking.
    if "SPY" not in tickers:
        tickers.append("SPY")
    
    print("Using composition from:", composition_at_start["date"].date())
    print(f"Found {len(tickers)} tickers in the composition.")
    
    # Download historical closing price data for each ticker.
    price_series_list = []
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start_date, end=end_date)
            if not df.empty:
                # Extract the 'Close' price series and set its name.
                s = df["Close"]
                s.name = ticker
                price_series_list.append(s)
            else:
                print(f"No data returned for {ticker}.")
        except Exception as e:
            print(f"Error downloading data for {ticker}: {e}")
    
    # Combine all individual series into one DataFrame.
    price_data_df = pd.concat(price_series_list, axis=1)
    
    # Save the downloaded data to CSV for future use.
    price_data_df.to_csv(data_csv)
    print(f"Downloaded data saved to {data_csv}")
    
    return price_data_df