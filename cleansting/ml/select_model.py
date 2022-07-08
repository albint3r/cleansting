from numpy import array
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge


def hpt_random_forest(X_train, y_train, **kwargs) -> object:
    """Apply a GridSearchCV to obtain the best Random Forest Model."""

    # Parameters
    n_estimators = kwargs.get('n_estimators', [int(x) for x in np.linspace(start=10, stop=80, num=10)])
    max_features = kwargs.get('max_features', ['auto', 'log2'])
    max_depth = kwargs.get('max_depth', [10, 12, 14, 16])
    min_samples_split = kwargs.get('min_samples_split', [2, 4])
    min_samples_leaf = kwargs.get('min_samples_leaf', [1, 2, 3])

    # Params Grid
    param_grid = {'n_estimators': n_estimators, 'max_features': max_features, 'max_depth': max_depth,
                  'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf}
    # Create Model
    model = RandomForestRegressor(random_state=42)
    # Create Grid
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, verbose=2, n_jobs=4).fit(X_train, y_train)

    # Fit and return
    return grid.best_estimator_


def hpt_ridge(X_train, y_train, X_test, y_test):
    model = Ridge()
    param_grid = {'alpha': np.arange(0, 100)}
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5).fit(X_train, y_train)
    grid = grid.best_estimator_
    y_pred = grid.predict(X_test)

    msg1 = f'Score: {grid.score(X_test, y_test)}'
    msg2 = f'Best Parameters: {grid}'
    print(msg1)
    print(msg2)

    return grid


def hpt_linear(X_train, y_train, X_test, y_test):
    model = LinearRegression()
    cv = cross_val_score(model, X_train, y_train, cv=5)

    # Fit Model
    model.fit(X_train, y_train)
    # Test Score
    pred_cv_score = cv.mean()
    pred_train_score = model.score(X_test, y_test)
    pred_test_score = model.score(X_test, y_test)

    msg1 = f'CV Score: {pred_cv_score:.2f}'
    msg2 = f'Train score {pred_train_score:.2f}'
    msg3 = f'Predict Score {pred_test_score:.2f}'
    msg4 = f'All Scores {np.mean([pred_cv_score, pred_train_score, pred_test_score]):.2f}'
    print(msg1)
    print(msg2)
    print(msg3)
    print('-----------------')
    print(msg4)

    return model.fit(X_train, y_train)


def compare_ml_models(data):
    df = data
    # Default Columns
    columns = ['m2_const', 'banos', 'autos']
    # Select Features
    X = df.loc[:, columns].values
    # Select Target
    y = df.price.values
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.20)
    # Creating Model
    model = LinearRegression()
    # Cross Val
    cross = cross_val_score(model, X_train, y_train, cv=10)
    # Model
    model.fit(X_train, y_train)
    # Predictions
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    # Var
    var_train = mean_squared_error(y_train, pred_train)
    var_test = mean_squared_error(y_test, pred_test)
    # Score
    score_train = model.score(X_train, y_train)
    score_test = model.score(X_test, y_test)
    # Diff
    error = y_test - pred_test

    msg = f'Cross Validation: {cross.mean():.2f}\n'
    msg1 = f'Var Train: {np.sqrt(var_train.mean()):,.0f}'
    msg2 = f'Score Train: {score_train.mean():,.2f}\n'
    msg3 = f'Var Test: {np.sqrt(var_test.mean()):,.0f}'
    msg4 = f'Score Test: {score_test.mean():,.2f}\n'
    msg5 = f'Error mean: {error.mean():,.2f}'
    msg6 = f'Error std: {error.std():,.2f}'

    print('------------------------')
    print(msg)
    print(msg1)
    print(msg2)
    print(msg3)
    print(msg4)
    print(msg5)
    print(msg6)


def per_error_predicts(y_true: array, y_pred: array) -> array:
    """Calculate the error of the prediction."""
    return np.abs((y_pred - y_true) / y_true) * 100


def error_resume(y_true: array, y_pred: array):
    error_predicts = per_error_predicts(y_true, y_pred)
    mean = error_predicts.mean()
    std = error_predicts.std()
    below_7 = (error_predicts <= 7).mean()

    msg = f'The mean error: {mean:.2f}%'
    msg1 = f'The std error: {std:.2f}'
    msg2 = f'Count Error below 7% : {below_7:.2f}%'

    print(msg)
    print(msg1)
    print(msg2)


def compare_model_error(X_test, y_test, **kwargs):
    models = kwargs

    for name, model in models.items():
        y_pred = model.predict(X_test)
        print(f'\n------{name.title()}--------')
        error_resume(y_test, y_pred)


def compare_score_models(X_train, X_test, y_train, y_test, **kwargs):
    """Create a test score comparation and select the model with best score."""

    models = kwargs

    best_model_score = list()

    for name, model in models.items():

        model.fit(X_train, y_train)
        # Train
        y_pred_train = model.score(X_train, y_train)
        y_train_pred = model.predict(X_train)
        av_error_train = (((y_train_pred - y_train) / y_train) * 100).mean()
        # Score
        y_pred_test = model.score(X_test, y_test)
        y_test_pred = model.predict(X_test)
        av_error_test = (((y_test_pred - y_test) / y_test) * 100).mean()
        # Add firt model
        if not best_model_score:
            best_model_score.append((name, y_pred_test, model))
        # Actualize the best model
        if best_model_score and best_model_score[0][1] < y_pred_test:
            best_model_score.pop()
            best_model_score.append((name, y_pred_test, model))

        # Print Area
        msg1 = f'Train Score: {y_pred_train:.2f}'
        msg2 = f'Mean Train error: {av_error_train:.2f}%'
        msg3 = f'Test Score: {y_pred_test:.2f}'
        msg4 = f'Mean Test error: {av_error_test:.2f}%'
        print(f'\n------- {name.title()} ---------')
        print(msg1)
        print(msg2, '\n')
        print(msg3)
        print(msg4)

    print('\n--------------------------------------')
    print(f'Best Model: {best_model_score[0][0].title()}')
    print(f'Score: {best_model_score[0][1] * 100:.0f}%')

    return best_model_score[0][2]