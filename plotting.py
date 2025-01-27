# plotting.py

import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates

def plot_pair_signals(bt_df, pair_label=""):
    """
    Given a DataFrame with relevant strategy columns, plot:
    - Stock A and B prices with buy/sell signals.
    - Exit signals (regular and due to max drawdown).
    - Strategy capital vs. SPY buy-and-hold performance.

    Parameters:
    - bt_df: DataFrame with columns ['A', 'B', 'Pos', 'Strategy_Capital', 'SPY_BuyHold', 'Beta', 
                                     'Raw_Return', 'Rolling_Volatility', 'Scaling_Factor', 'Adjusted_Return']
    - pair_label: String label for the pair (e.g., "AAPL vs. MSFT")
    """
    # Ensure the DataFrame is sorted by date
    bt_df = bt_df.sort_index()

    # Identify position changes
    bt_df['Prev_Pos'] = bt_df['Pos'].shift(1).fillna(0)
    bt_df['Position_Change'] = bt_df['Pos'] - bt_df['Prev_Pos']

    # Entry and Exit points
    entry_long = bt_df[bt_df['Position_Change'] == 1]
    entry_short = bt_df[bt_df['Position_Change'] == -1]
    exit_regular = bt_df[(bt_df['Position_Change'] == -1) & (bt_df['Prev_Pos'] == 1) |
                         (bt_df['Position_Change'] == 1) & (bt_df['Prev_Pos'] == -1)]
    
    # Exits due to max drawdown
    exit_max_drawdown = bt_df[bt_df['Exit_DD'] == True] if 'Exit_DD' in bt_df.columns else pd.DataFrame()

    # Create a figure
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot Stock A and Stock B
    ax.plot(bt_df.index, bt_df['A'], label='Stock A', color='blue')
    ax.plot(bt_df.index, bt_df['B'], label='Stock B', color='orange')

    # Plot Buy/Sell Signals for Stock A
    ax.scatter(entry_long.index, bt_df.loc[entry_long.index, 'A'], marker='^', color='green', s=100, label='Buy A')
    ax.scatter(entry_short.index, bt_df.loc[entry_short.index, 'A'], marker='v', color='red', s=100, label='Sell A')

    # Plot Buy/Sell Signals for Stock B
    ax.scatter(entry_short.index, bt_df.loc[entry_short.index, 'B'], marker='^', color='green', s=100, label='Buy B')
    ax.scatter(entry_long.index, bt_df.loc[entry_long.index, 'B'], marker='v', color='red', s=100, label='Sell B')

    # Plot Exit Signals (Regular Exit)
    ax.scatter(exit_regular.index, bt_df.loc[exit_regular.index, 'A'], marker='x', color='black', s=40, label='Exit Regular')
    ax.scatter(exit_regular.index, bt_df.loc[exit_regular.index, 'B'], marker='x', color='black', s=40)

    # Plot Exit Signals (Max Drawdown)
    if not exit_max_drawdown.empty:
        ax.scatter(exit_max_drawdown.index, bt_df.loc[exit_max_drawdown.index, 'A'], marker='x', color='purple', s=40, label='Exit Max Drawdown')
        ax.scatter(exit_max_drawdown.index, bt_df.loc[exit_max_drawdown.index, 'B'], marker='x', color='purple', s=40)

    # Set labels and title
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.set_title(f"Pair Trading Signals: {pair_label}")

    # Format x-axis to show years
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=45)

    # Combine legends and remove duplicates
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left')

    # Edit layout
    ax.grid(True)

    plt.tight_layout()
    plt.show()
