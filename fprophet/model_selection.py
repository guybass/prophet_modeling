from fbprophet import Prophet
from sklearn.model_selection import ParameterGrid
import numpy as np

def tune_prophet_hyperparameters(df, params_grid):
    """
    Tunes hyperparameters of a Prophet model using grid search.

    Args:
        df: pandas DataFrame containing the time series
        params_grid: a dictionary specifying the hyperparameters to tune and their values to try

    Returns:
        The best Prophet model found during grid search.
    """
    # Generate all possible combinations of hyperparameters
    param_grid = ParameterGrid(params_grid)

    # Initialize variables to keep track of best model and its score
    best_score = np.inf
    best_model = None

    # Iterate over each combination of hyperparameters
    for params in param_grid:
        # Initialize a Prophet model with the given hyperparameters
        model = Prophet(**params)

        # Fit the model to the data
        model.fit(df)

        # Get the cross-validation scores of the model
        cross_val_scores = model.cross_validation_horizon

        # Calculate the mean squared error of the model
        score = np.mean(cross_val_scores['mse'])

        # Update the best model and score if this model is better
        if score < best_score:
            best_score = score
            best_model = model

    # Return the best model found during grid search
    return best_model
