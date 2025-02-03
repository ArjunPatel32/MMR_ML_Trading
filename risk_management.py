# risk_management.py

def apply_stop_loss(final_signals, price_data, stop_loss_pct=0.05):
    """
    Applies a daily stop-loss rule to the existing final_signals:
      - If a long position loses more than stop_loss_pct, we close it that day (signal=0).
      - If a short position loses more than stop_loss_pct, we close it that day (signal=0).
    
    final_signals : DataFrame of {-1,0,+1}, index=dates, columns=tickers
    price_data    : DataFrame of prices, same shape or superset
    stop_loss_pct : e.g. 0.05 for 5% max adverse move
    
    Returns a modified copy of final_signals with stop-loss enforced.
    """
    signals_sl = final_signals.copy()
    prices = price_data.reindex(signals_sl.index, columns=signals_sl.columns).ffill()

    # Store the entry price + side for each ticker
    entry_price = {}
    entry_side  = {}

    # Ensure dates are in ascending order
    dates = signals_sl.index.sort_values()

    for i, date in enumerate(dates):
        row_signals = signals_sl.loc[date]

        for ticker, signal in row_signals.items():
            px = prices.at[date, ticker]

            # Look at the previous day's signal for that ticker
            if i > 0:
                prev_signal = signals_sl[ticker].iloc[i-1]
            else:
                prev_signal = 0

            # Check if a new position was opened today
            if signal != 0 and prev_signal == 0:
                # record entry
                entry_price[ticker] = px
                entry_side[ticker]  = signal

            # If we are still in a position, check stop-loss
            if signal != 0:
                side = entry_side.get(ticker, signal)
                ep   = entry_price.get(ticker, px)

                # If we're long, and px <= ep*(1 - stop_loss_pct), close
                if side == 1 and px <= ep * (1 - stop_loss_pct):
                    signals_sl.at[date, ticker] = 0
                # If we're short, and px >= ep*(1 + stop_loss_pct), close
                elif side == -1 and px >= ep * (1 + stop_loss_pct):
                    signals_sl.at[date, ticker] = 0

            # If signal == 0, that means we’re not holding a position now
            # so next day, if we reopen, new entry price will be recorded.
    
    return signals_sl
