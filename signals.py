# signals.py

def generate_signals_momentum(momentum_df, threshold=0.05):
    """
    Returns a +1/-1 signal for each ticker/day if momentum is above/below 'threshold'.
    0 otherwise.
    """
    signals = momentum_df.copy()
    signals[:] = 0  # initialize

    signals[momentum_df > threshold] = 1
    signals[momentum_df < -threshold] = -1

    return signals

def generate_signals_meanreversion(zscore_df, z_entry=1.0):
    """
    Returns a +1 (long) signal if zscore < -z_entry,
           a -1 (short) signal if zscore > z_entry,
           and 0 if inside the neutral zone (|zscore| < z_exit).
    """
    signals = zscore_df.copy()
    signals[:] = 0  # initialize

    signals[zscore_df < -z_entry] = 1
    signals[zscore_df > z_entry] = -1
    return signals

def generate_final_signal(clf, features, momentum_signals, meanrev_signals):
    """
    For each day, if clf predicts 1 -> use momentum_signals,
                     if clf predicts 0 -> use meanrev_signals.
    Returns a DataFrame with the chosen signals day by day.
    """
    # Predict
    X = features.loc[momentum_signals.index]  # ensure same dates
    # For days we have missing features, fill or drop
    X = X.ffill().replace({None: 0})

    predictions = clf.predict(X)

    # Combine
    final_signals = momentum_signals.copy()
    for i, date in enumerate(final_signals.index):
        if predictions[i] == 0:
            # use mean-reversion signals for this day
            final_signals.loc[date] = meanrev_signals.loc[date]
        else:
            # use momentum signals
            final_signals.loc[date] = momentum_signals.loc[date]

    return final_signals
