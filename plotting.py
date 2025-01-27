# plotting.py

import matplotlib.pyplot as plt

def plot_pair_signals(bt_df, pair_label=""):
    """
    Given a DataFrame with columns ['A','B','Pos'], 
    plot each stock's price with markers showing buy/sell signals.
    """
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(12,8))

    # Extract buy/sell indices
    buy_idx = bt_df[bt_df['Pos'] == 1].index   # Strategy is long A, short B
    sell_idx = bt_df[bt_df['Pos'] == -1].index # Strategy is short A, long B

    # Subplot 1: Stock A with signals

    ax1.plot(bt_df.index, bt_df['A'], label='Stock A', color='blue')
    # Buy signals on Stock A => green ^
    ax1.scatter(buy_idx, bt_df.loc[buy_idx, 'A'], 
                marker='^', color='green', alpha=0.8, label='Buy/Long A')
    # Sell signals on Stock A => red v
    ax1.scatter(sell_idx, bt_df.loc[sell_idx, 'A'],
                marker='v', color='red', alpha=0.8, label='Sell/Short A')
    
    ax1.set_ylabel('Price ($)')
    ax1.legend(loc='best')
    ax1.set_title(f"Stock A: {pair_label}")

    # Subplot 2: Stock B with signals

    ax2.plot(bt_df.index, bt_df['B'], label='Stock B', color='orange')
    # When we are long A (Pos=1), we are short B => mark red 'v'
    ax2.scatter(buy_idx, bt_df.loc[buy_idx, 'B'], 
                marker='v', color='red', alpha=0.8, label='Short B (when Pos=1)')
    # When we are short A (Pos=-1), we are long B => mark green '^'
    ax2.scatter(sell_idx, bt_df.loc[sell_idx, 'B'],
                marker='^', color='green', alpha=0.8, label='Long B (when Pos=-1)')

    ax2.set_ylabel('Price ($)')
    ax2.set_title(f"Stock B: {pair_label}")
    ax2.legend(loc='best')

    plt.tight_layout()
    plt.show()