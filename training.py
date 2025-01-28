from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier

def train_strategy_chooser(features, label):
    """
    Train a binary classifier that predicts whether
    momentum (1) or mean-reversion (0) will be better next day,
    using time-series cross validation and grid search.
    """

    # Align features & label
    X, y = features.align(label, join='inner', axis=0)
    # Drop Nan values
    X = X.dropna()
    y = y.loc[X.index]

    # Time-series cross-validation, 4 splits
    tscv = TimeSeriesSplit(n_splits=4)

    # Define a parameter grid
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 3, 5],
        'min_samples_split': [2, 5],
    }

    # Create the RandomForest and GridSearch
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=tscv,
        scoring='accuracy',  # or 'f1', 'roc_auc', etc.
        n_jobs=-1
    )

    # 5) Fit the grid search
    grid_search.fit(X, y)

    # 6) Best model & best params
    best_model = grid_search.best_estimator_
    print("Best Params:", grid_search.best_params_)
    print("Best CV Score:", grid_search.best_score_)

    return best_model

def train_momentum_classifier(momentum, future_returns):
    features = momentum.shift(1)
    target = (future_returns > 0).astype(int)

    # Align features and target to the same dates
    features, target = features.align(target, join='inner')

    # Drop any remaining NaNs
    features = features.dropna()
    target = target.loc[features.index]

    X_train, X_test, y_train, y_test = train_test_split(
        features, 
        target, 
        test_size=0.2, 
        random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model