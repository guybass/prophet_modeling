from typing import Dict, Any, List, Tuple
import pandas as pd
from fbprophet import Prophet, diagnostics
import matplotlib.pyplot as plt


def plot_prophet_forecast(df: pd.DataFrame, time_col: str, value_col: str, freq: str,
                          forecast_period: int, changepoint_prior_scale: float = 0.05,
                          yearly_seasonality: bool = True, weekly_seasonality: bool = True,
                          daily_seasonality: bool = True) -> None:
    """
    Fits a Prophet model to the provided DataFrame and plots the observed data, the fitted model, and the predicted
    values.

    Args:
        df (pd.DataFrame): A pandas DataFrame containing the time series data to fit and forecast.
        time_col (str): The name of the column containing the time information.
        value_col (str): The name of the column containing the values to forecast.
        freq (str): The frequency of the time series data. Can be any valid Pandas frequency string, e.g. 'D' for daily,
            'M' for monthly.
        forecast_period (int): The number of periods to forecast.
        changepoint_prior_scale (float, optional): Parameter regulating the flexibility of the automatic changepoint
            selection. Increase it to make the trend more flexible. Defaults to 0.05.
        yearly_seasonality (bool, optional): Whether to include yearly seasonality in the model. Defaults to True.
        weekly_seasonality (bool, optional): Whether to include weekly seasonality in the model. Defaults to True.
        daily_seasonality (bool, optional): Whether to include daily seasonality in the model. Defaults to True.

    Returns:
        None.

    Raises:
        ValueError: If the provided DataFrame is empty or if the provided value column does not exist.

    """

    if df.empty:
        raise ValueError("The provided DataFrame is empty.")

    if value_col not in df.columns:
        raise ValueError(f"The provided value column '{value_col}' does not exist.")

    # Rename the columns to the names expected by Prophet
    df = df.rename(columns={time_col: 'ds', value_col: 'y'})

    # Initialize Prophet model
    model = Prophet(
        changepoint_prior_scale=changepoint_prior_scale,
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=daily_seasonality,
    )

    # Fit the model to the data
    model.fit(df)

    # Make a DataFrame with future dates
    future = model.make_future_dataframe(periods=forecast_period, freq=freq)

    # Generate the forecast
    forecast = model.predict(future)

    # Plot the forecast
    fig = model.plot(forecast)

    # Set plot labels
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('Observed Data, Fitted Model, and Forecast')

    # Show the plot
    plt.show()


def plot_prophet_components(df: pd.DataFrame, plot_components: bool = True,
                            plot_weekly: bool = True, plot_yearly: bool = True) -> None:
    """
    Fits a Prophet model to the provided DataFrame and plots the observed data, the fitted model, and the predicted values,
    along with their components.

    Args:
        df: pandas DataFrame containing the time series
        plot_components: flag to determine if individual component plots should be included (default True)
        plot_weekly: flag to determine if weekly component plot should be included (default True)
        plot_yearly: flag to determine if yearly component plot should be included (default True)

    Returns:
        None

    Mathematical explanation:
    Prophet is a decomposable time series model with three main components: trend, seasonality, and holidays. In addition,
    it allows for including user-specified regressors. This function uses Prophet to fit a model to the provided DataFrame and
    then plots the trend component, seasonality components (weekly and/or yearly, if selected), and any additional regressors
    as specified in the DataFrame.
    """

    # Instantiate Prophet object and fit to the data
    model = Prophet()
    model.fit(df)

    # Generate forecast for future periods
    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)

    # Plot the forecast
    fig = model.plot(forecast)

    # Add components to the forecast plot, if desired
    if plot_components:
        components = model.plot_components(forecast)

        # Add weekly component plot, if desired
        if plot_weekly:
            weekly = model.plot_weekly(forecast)
            weekly.set_size_inches(12, 8)
            plt.show()

        # Add yearly component plot, if desired
        if plot_yearly:
            yearly = model.plot_yearly(forecast)
            yearly.set_size_inches(12, 8)
            plt.show()

    plt.show()


def plot_seasonality(ts_data: pd.DataFrame, freq: str, holidays: List[Tuple[str, str]], yearly_start: int = 0) -> None:
    """
    Plots classic seasonality of a time series with the yearly, weekly, and daily seasonality components.

    Args:
        ts_data (pd.DataFrame): Pandas DataFrame containing the time series data. It must contain a 'ds' column with the dates and a 'y' column with the values.
        freq (str): The frequency of the time series, such as 'D' for daily, 'W' for weekly, or 'M' for monthly.
        holidays (List[Tuple[str, str]]): List of tuples representing the start and end dates of holidays. The dates must be in 'YYYY-MM-DD' format.
        yearly_start (int, optional): The starting year of the yearly seasonality plot. Defaults to 0.

    Returns:
        None.

    Mathematical explanation:
    The function uses Prophet to fit the time series and calculate the yearly, weekly, and daily seasonality components of the time series. It then plots these components using the plot_yearly, plot_weekly, and plot_daily functions of Prophet.
    """

    # Initialize and fit the Prophet model
    model = Prophet(holidays=holidays, yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
    model.fit(ts_data)

    # Make future predictions and create a forecast dataframe
    future = model.make_future_dataframe(periods=365, freq=freq)
    forecast = model.predict(future)

    # Plot the forecast with the yearly, weekly, and daily seasonality components
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(15, 10))
    model.plot(forecast, ax=axs[0])
    model.plot_yearly(ax=axs[1], uncertainty=False, plot_cap=False, xlabel='Year', year_start=yearly_start)
    model.plot_weekly(ax=axs[2], uncertainty=False, plot_cap=False, xlabel='Day of Week')
    model.plot_daily(ax=axs[2], uncertainty=False, plot_cap=False, xlabel='Time of Day')

    # Display the plot
    plt.show()


def plot_prophet_cv_metrics(model: Prophet, df: pd.DataFrame, metric: str = 'mape',
                            rolling_window: int = None, horizon: str = '365 days') -> None:
    """
    Plots the cross validation metrics for a Prophet model.

    Args:
        model (Prophet): A fitted Prophet model
        df (pd.DataFrame): A pandas DataFrame containing the time series data
        metric (str): The metric to be used for evaluation during cross-validation
        rolling_window (int): The size of the rolling window to use for the metric calculation
        horizon (str): The forecast horizon to use for cross-validation
    """

    cv_results = diagnostics.cross_validation(model=model, horizon=horizon, rolling_window=rolling_window)
    cv_metrics = diagnostics.performance_metrics(cv_results, metric=metric, rolling_window=rolling_window)

    if rolling_window is not None:
        title = f"{metric} ({rolling_window}-day rolling window)"
    else:
        title = f"{metric} (overall)"

    fig = model.plot_cross_validation_metric(metric=metric, rolling_window=rolling_window)

    fig.suptitle(title)


def get_cv_scores(df: pd.DataFrame, model: Prophet, horizon: str = '365 days', period: str = '180 days',
                  initial: str = '730 days', parallel: str = 'processes') -> pd.DataFrame:
    """
    Calculates cross-validation scores for the given Prophet model and time series data.

    Parameters:
    -----------
    df : pandas.DataFrame
        The time series data as a DataFrame with 'ds' and 'y' columns.
    model : Prophet object
        The Prophet model to use for cross-validation.
    horizon : str, optional
        The forecast horizon. The default is '365 days'.
    period : str, optional
        The spacing between cutoff dates. The default is '180 days'.
    initial : str, optional
        The size of the initial training period. The default is '730 days'.
    parallel : str, optional
        Method for parallelizing the computation. The default is 'processes'.

    Returns:
    --------
    pandas.DataFrame
        The cross-validation scores as a DataFrame with 'cutoff', 'horizon', and 'mse' columns.
    """
    cv_results = cross_validation(model, horizon=horizon, period=period, initial=initial, parallel=parallel, data=df)
    return cv_results