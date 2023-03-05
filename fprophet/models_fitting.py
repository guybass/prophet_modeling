import pandas as pd
from typing import Dict, List, Tuple, Union
from fbprophet import Prophet


def fit_linear_trend(df: pd.DataFrame) -> Prophet:
    """
    Fits a Prophet model with a linear trend to the provided DataFrame.
    ----------------

    Args:
        df: pandas DataFrame containing the time series
    ----------------

    Returns:
        fitted Prophet model
    ----------------

    Mathematical explanation:
    Linear Trend Model:
    The linear trend model in Prophet is a special case of the additive model, where the trend component, g(t), is modeled as a linear function of time. This can be expressed as:
    g(t) = k + mt

    where:

        k represents the intercept of the trend
        m represents the slope of the trend
        In the linear trend model, the other components of the additive model, including seasonality and holidays, are also included. The formula for the linear trend model can be expressed as:

    y(t) = k + mt + s(t) + h(t) + e(t)

    where y(t), s(t), h(t), and e(t) have the same meanings as in the additive model.
    """
    m = Prophet()
    m.fit(df)
    return m


def fit_additive_seasonality(df: pd.DataFrame, seasonality_prior_scale=10.0) -> Prophet:
    """
    Fits a Prophet model with additive seasonality to the provided DataFrame.
    ----------------

    Args:
        df: pandas DataFrame containing the time series
    ----------------

    Returns:
        fitted Prophet model
    ----------------

    Mathematical explanation:
    Additive Seasonality Model:
    The additive seasonality model in Prophet is a special case of the additive model.
    where the seasonality component, s(t), is modeled as a set of periodic functions. This can be expressed as:
    s(t) = s1(t) + s2(t) + ... + sn(t)
    """
    m = Prophet(seasonality_mode='additive', seasonality_prior_scale=seasonality_prior_scale)
    m.fit(df)
    return m


def fit_multiplicative_seasonality(df: pd.DataFrame, seasonality_prior_scale=10.0) -> Prophet:
    """
    Fits a Prophet model with multiplicative seasonality to the provided DataFrame.
    ----------------

    Args:
        df: pandas DataFrame containing the time series
    ----------------

    Returns:
        fitted Prophet model
    ----------------

    Mathematical explanation:
    Multiplicative Seasonality Model:
    The multiplicative seasonality model in Prophet is a variation of the additive model, where the seasonal component. \
    s(t), is modeled as a multiplicative factor rather than an additive term. This can be expressed as:
    y(t) = g(t) * s(t) * h(t) * e(t)

    where the terms have the same meanings as in the additive model.
    """
    m = Prophet(seasonality_mode='multiplicative', seasonality_prior_scale=seasonality_prior_scale)
    m.fit(df)
    return m


def fit_fourier_series_seasonality(df: pd.DataFrame, fourier_order=10) -> Prophet:
    """
    Fits a Prophet model with seasonality modeled using Fourier series to the provided DataFrame.
    ----------------

    Args:
        df: pandas DataFrame containing the time series
    ----------------

    Returns:
        fitted Prophet model
    ----------------

    Mathematical explanation:
    Seasonality Model with Fourier Series:
    The seasonality model with Fourier series in Prophet is another variation of the additive model, where the seasonal component, s(t), is modeled using a Fourier series rather than a fixed set of seasonal indicators. This can be expressed as:
    s(t) = ∑(i=1)^n [a(i)cos(2πit/P) + b(i)sin(2πit/P)]

    where:

        n is the number of Fourier terms used to model the seasonality
        a(i) and b(i) are the coefficients of the Fourier terms
        P is the length of the seasonality period (e.g., 7 for weekly seasonality, 365.25 for yearly seasonality)
        In the seasonality model with Fourier series, the seasonal component is modeled as a combination of sine and cosine functions with different frequencies and amplitudes.
        This allows the model to capture complex seasonal patterns with multiple frequencies. The other components of the additive model, including trend and holidays, are also included.
        The formula for the seasonality model with Fourier series can be expressed as:

        y(t) = g(t) + ∑(i=1)^n [a(i)cos(2πit/P) + b(i)sin(2πit/P)] + h(t) + e(t)

        where the terms have the same meanings as in the additive model.
    """
    m = Prophet()
    m.add_seasonality(name='fourier', period=365.25, fourier_order=fourier_order)
    m.fit(df)
    return m


def fit_holidays(df: pd.DataFrame, holidays: pd.DataFrame) -> Prophet:
    """
    Fits a Prophet model with holiday effects to the provided DataFrame.
    ----------------

    Args:
        df: pandas DataFrame containing the time series
        holidays: pandas DataFrame containing the holidays
    ----------------

    Returns:
        fitted Prophet model
    ----------------

    Mathematical explanation:
    Holidays Model:
    The holidays model in Prophet is an extension of the additive model that allows for the inclusion of user-defined holidays.
    In this model, the holiday component, h(t), is a binary indicator variable that takes on a value of 1 for each holiday period and 0 otherwise.
    The formula for the holidays model can be expressed as:
    y(t) = g(t) + s(t) + h(t) + e(t)

    where y(t), g(t), s(t), and e(t) have the same meanings as in the additive model.

    In the holidays model, the user can provide a list of dates corresponding to holidays that may have an impact on the time series.
    The model then includes a separate binary indicator variable for each holiday period, which allows for the estimation of the impact of each holiday on the time series.
    """
    m = Prophet(holidays=holidays)
    m.fit(df)
    return m


def fit_custom_seasonality(df: pd.DataFrame, custom_seasonality: Dict, seasonality_prior_scale=10.0) -> Prophet:
    """
    Fits a Prophet model with custom seasonality to the provided DataFrame.
    ----------------

    Args:
        df: pandas DataFrame containing the time series
        custom_seasonality: dictionary containing the custom seasonality
        seasonality_prior_scale: hyperparameter controlling the flexibility of the seasonality model (default 10.0)
    ----------------

    Returns:
        fitted Prophet model
    ----------------

    Mathematical explanation:
    Custom Seasonality Model:
    The custom seasonality model in Prophet allows for the inclusion of user-defined seasonality patterns that do not follow a fixed frequency or periodicity.
    In this model, the seasonal component, s(t), is modeled using a set of Fourier terms, similar to the seasonality model with Fourier series.
    However, in the custom seasonality model, the frequencies and amplitudes of the Fourier terms are not fixed, and can be specified by the user.
    The formula for the custom seasonality model can be expressed as:
    y(t) = g(t) + ∑(i=1)^n [a(i)cos(2πf(i)*t) + b(i)sin(2πf(i)*t)] + h(t) + e(t)

    where:

    n is the number of Fourier terms used to model the seasonality
    a(i) and b(i) are the coefficients of the Fourier terms
    f(i) is the frequency of the i-th Fourier term
    In the custom seasonality model, the user can provide a list of dates and associated frequencies, which allows the model to capture any custom seasonality patterns that may be present in the data.
    """
    m = Prophet(seasonality_mode='multiplicative', seasonality_prior_scale=seasonality_prior_scale)
    for name, period in custom_seasonality.items():
        m.add_seasonality(name=name, period=period)
    m.fit(df)
    return m


def fit_growth_model(df: pd.DataFrame, growth: str, changepoint_prior_scale=0.05, **kwargs) -> Prophet:
    """
    Fits a Prophet model with logistic or linear growth to the provided DataFrame.
    ----------------

    Args:
        df: pandas DataFrame containing the time series
        growth: type of growth ('linear' or 'logistic')
        changepoint_prior_scale: hyperparameter controlling flexibility of the automatic changepoint selection
        kwargs: additional arguments to pass to Prophet
    ----------------

    Returns:
        fitted Prophet model
    ----------------

    Mathematical explanation:
    Growth Model (with logistic or linear growth):
    The growth model in Prophet allows for the inclusion of a growth term in the trend component, g(t).
    The growth term can be modeled as either a linear function or a logistic function of time.
    The formula for the growth model with linear growth can be expressed as:
    g(t) = k + mt

    where k represents the intercept of the trend, and m represents the slope of the trend.
    The formula for the growth model with logistic growth can be expressed as:

    g(t) = c / (1 + e^(-k-m*t))

    where c represents the carrying capacity of the growth, and k and m represent the parameters of the logistic growth curve.
    """

    m = Prophet(growth=growth, changepoint_prior_scale=changepoint_prior_scale, **kwargs)
    m.fit(df)
    return m


def fit_saturating_minimum_model(df: pd.DataFrame, floor: float, changepoint_prior_scale=0.05, **kwargs) -> Prophet:
    """
    Fits a Prophet model with a saturating minimum to the provided DataFrame.
    ----------------

    Args:
        df: pandas DataFrame containing the time series
        floor: lower bound on the predicted values
        changepoint_prior_scale: hyperparameter controlling flexibility of the automatic changepoint selection
        kwargs: additional arguments to pass to Prophet
    ----------------

    Returns:
        fitted Prophet model
    ----------------

    Mathematical explanation:
    Saturating Minimum Model:
    The saturating minimum model in Prophet is a variation of the growth model that includes a minimum limit on the value of the time series.
    In this model, the trend component, g(t), is modeled using a saturating exponential function, which approaches a minimum limit over time.
    The formula for the saturating minimum model can be expressed as:
    g(t) = k + (m/(1 + e^(-r*(t-t0)))) + B*(t-t0)

    where:

        k represents the intercept of the trend
        m represents the initial slope of the trend
        r represents the rate of saturation of the trend
        t0 represents the inflection point of the trend
        B represents the slope of the linear trend component
        The saturating minimum model allows the user to specify a minimum limit on the value
    """
    m = Prophet(seasonality_mode='multiplicative', growth='logistic', changepoint_prior_scale=changepoint_prior_scale, **kwargs)
    m.add_floor(name='floor', value=floor)
    m.fit(df)
    return m


def fit_saturating_maximum_model(df: pd.DataFrame, ceiling: float, changepoint_prior_scale=0.05, **kwargs) -> Prophet:
    """
    Fits a Prophet model with a saturating maximum to the provided DataFrame.
    ----------------

    Args:
        df: pandas DataFrame containing the time series
        ceiling: upper bound on the predicted values
        changepoint_prior_scale: hyperparameter controlling flexibility of the automatic changepoint selection
        kwargs: additional arguments to pass to Prophet
    ----------------

    Returns:
        fitted Prophet model
    ----------------

    Mathematical explanation:
    Saturating Maximum Model:
    The saturating maximum model in Prophet is a variation of the saturating minimum model that includes a maximum limit on the value of the time series. In this model, the trend component, g(t), is modeled using a saturating logistic function, which approaches a maximum limit over time. The formula for the saturating maximum model can be expressed as:
    g(t) = k + (m/(1 + e^(-r*(t-t0)))) + B*(t-t0)

    where:

        k represents the intercept of the trend
        m represents the initial slope of the trend
        r represents the rate of saturation of the trend
        t0 represents the inflection point of the trend
        B represents the slope of the linear trend component
    The saturating maximum model allows the user to specify a maximum limit on the value of the time series.
    """
    m = Prophet(seasonality_mode='multiplicative', growth='logistic', changepoint_prior_scale=changepoint_prior_scale, **kwargs)
    m.add_ceiling(name='ceiling', value=ceiling)
    m.fit(df)
    return m


def fit_changepoints_model(df: pd.DataFrame, change_points: tuple(list[str], pd.DatetimeIndex),
                           changepoint_prior_scale=0.05, **kwargs) -> Prophet:
    """
    Fits a Prophet model with manually specified changepoints to the provided DataFrame.
    ----------------

    Args:
        df: pandas DataFrame containing the time series
        change_points: List of dates at which to include potential changepoints
        changepoint_prior_scale: hyperparameter controlling flexibility of the automatic changepoint selection
        kwargs: additional arguments to pass to Prophet
    ----------------

    Returns:
        fitted Prophet model
    ----------------

    Mathematical explanation:
    Change_points Model:
    The change_points model in Prophet allows for the inclusion of abrupt changes or "changepoints" in the trend component, g(t).
    In this model, the trend component is divided into a series of piecewise linear segments, with each segment corresponding to a period of relatively stable trend.
    The formula for the changepoints model can be expressed as:
    y(t) = g(t) + s(t) + h(t) + e(t)

    where y(t), s(t), h(t), and e(t) have the same meanings as in the additive model, and g(t) is defined as:

    g(t) = k + ∑(i=1)^n (δ(i) * max(0, t - t(i)))

    where:

        k represents the intercept of the trend
        δ(i) represents the change in the trend at the i-th changepoint
        t(i) represents the time of the i-th changepoint
        n represents the number of changepoints in the model
        The changepoints model allows the user to specify the number and locations of the changepoints, which allows the model to capture any abrupt changes in the trend component.
    """
    m = Prophet(changepoints=change_points, changepoint_prior_scale=changepoint_prior_scale, **kwargs)
    m.fit(df)
    return m


def fit_trend_piecewise_linear(df: pd.DataFrame, changepoint_dates: Union[List[str], pd.DatetimeIndex], changepoint_prior_scale=0.05, **kwargs) -> Prophet:
    """
    Fits a Prophet model with trend using piecewise linear models to the provided DataFrame.
    ----------------

    Args:
        df: pandas DataFrame containing the time series
        changepoint_dates: List of dates at which to include potential changepoints
        changepoint_prior_scale: hyperparameter controlling flexibility of the automatic changepoint selection
        kwargs: additional arguments to pass to Prophet
    ----------------

    Returns:
        fitted Prophet model
    ----------------

    Mathematical explanation:
    Trend Piecewise Linear Model:
    Trend with Piecewise Linear Models:
    The trend with piecewise linear models in Prophet is a variation of the changepoints model that allows for the inclusion of additional regressors in the trend component, g(t).
    In this model, the trend component is divided into a series of piecewise linear segments, with each segment corresponding to a period of relatively stable trend.
    The formula for the trend with piecewise linear models can be expressed as:
    y(t) = g(t, x) + s(t) + h(t) + e(t)

    where y(t), s(t), h(t), and e(t) have the same meanings as in the additive model, and g(t, x) is defined as:

    g(t, x) = k + ∑(i=1)^n (δ(i) * max(0, t - t(i))) + β(x)

    where:

        k represents the intercept of the trend
        δ(i) represents the change in the trend at the i-th changepoint
        t(i) represents the time of the i-th changepoint
        n represents the number of changepoints in the model
        x represents the additional regressor(s) included in the trend component
        β represents the coefficients of the additional regressor(s)
        The trend with piecewise linear models allows the user to include additional regressors in the trend component, which may improve
    """
    model = Prophet(changepoint_prior_scale=changepoint_prior_scale, **kwargs)
    for date in changepoint_dates:
        model.add_seasonality(name=f"piecewise_linear_{date}", period=1, fourier_order=1, prior_scale=0.1)
    model.fit(df)
    return model


def fit_non_daily_data(df: pd.DataFrame, seasonality_mode: str = "multiplicative", **kwargs) -> Prophet:
    """
    Fits a Prophet model to non-daily time series data.
    ----------------

    Args:
        df: pandas DataFrame containing the time series
        seasonality_mode: seasonality mode to use (e.g. "multiplicative")
        kwargs: additional arguments to pass to Prophet
    ----------------

    Returns:
        fitted Prophet model
    ----------------

    Mathematical explanation:
    Non-daily Data Model:
    The non-daily data model in Prophet is designed to handle time series data that is irregularly spaced or has missing values.
    In this model, the trend component, g(t), is modeled using a piecewise linear model, and the seasonality component, s(t), is modeled using a Fourier series.
    The formula for the non-daily data model can be expressed as:
    y(t) = g(t) + s(t) + h(t) + e(t)

    where y(t), s(t), h(t), and e(t) have the same meanings as in the additive model, and g(t) is defined as:

    g(t) = k + ∑(i=1)^n (δ(i) * max(0, t - t(i)))

    where:

        k represents the intercept of the trend
        δ(i) represents the change in the trend at the i-th changepoint
        t(i) represents the time of the i-th changepoint
        n represents the number of changepoints in the model
    The non-daily data model allows the user to handle missing values or irregularly spaced data by modeling the trend component as a piecewise linear function.
    """
    model = Prophet(seasonality_mode=seasonality_mode, **kwargs)
    model.fit(df)
    return model


def fit_multiple_seasonality(df: pd.DataFrame, seasonality_modes: List[str], **kwargs) -> Prophet:
    """
    Fits a Prophet model with multiple seasonality components to the provided DataFrame.
    ----------------

    Args:
        df: pandas DataFrame containing the time series
        seasonality_modes: list of seasonality modes to use (e.g. ["multiplicative", "additive"])
        kwargs: additional arguments to pass to Prophet
    ----------------

    Returns:
        fitted Prophet model
    ----------------

    Mathematical explanation:
    Multiple Seasonality Model:
    The multiple seasonality model in Prophet is designed to handle time series data with multiple seasonal patterns.
    In this model, the seasonality component, s(t), is modeled using Fourier series for each seasonal pattern.
    The formula for the multiple seasonality model can be expressed as:
    y(t) = g(t) + ∑(i=1)^n s(i)(t) + h(t) + e(t)

    where y(t), g(t), h(t), and e(t) have the same meanings as in the additive model, and s(i)(t) represents the seasonality component for the i-th seasonal pattern.

    The multiple seasonality model allows the user to model time series data with multiple seasonal patterns, such as weekly and yearly patterns.
    """
    model = Prophet(**kwargs)
    for mode in seasonality_modes:
        model.add_seasonality(name=mode, period=1, fourier_order=5, prior_scale=0.02)
    model.fit(df)
    return model


def fit_dynamic_regressor(df: pd.DataFrame, regressors: pd.DataFrame, **kwargs) -> Prophet:
    """
    Fits a Prophet model with dynamic regressors to the provided DataFrame.
    ----------------

    Args:
        df: pandas DataFrame containing the time series
        regressors: pandas DataFrame containing the dynamic regressors
        kwargs: additional arguments to pass to Prophet
    ----------------

    Returns:
        fitted Prophet model
    ----------------

    Mathematical explanation:
    Dynamic Regressor Model:
    The dynamic regressor model in Prophet is designed to handle time series data with additional regressors that may influence the value of the time series.
    In this model, the trend component, g(t), is modeled using a piecewise linear model, and the seasonality component, s(t), is modeled using a Fourier series.
    The formula for the dynamic regressor model can be expressed as:
    y(t) = g(t, x) + s(t) + h(t) + e(t)

    where y(t), s(t), h(t), and e(t) have the same meanings as in the additive model, and g(t, x) is defined as:

    g(t, x) = k + ∑(i=1)^n (δ(i) * max(0, t - t(i))) + β(x)

    where:

    k represents the intercept of the trend
    δ(i) represents the change in the trend at the i-th changepoint
    t(i) represents the time of the i-th changepoint
    n represents the number of changepoints in the model
    x represents the additional regressor(s) that may influence the value of the time series
    β represents the coefficients of the additional regressor(s)
    The dynamic regressor model allows the user to include additional regressors in the model, which may improve the accuracy of the forecasts.
    """
    model = Prophet(**kwargs)
    for col in regressors.columns:
        model.add_regressor(col)
    model.fit(pd.concat([df, regressors], axis=1))
    return model


def fit_ar_model(df: pd.DataFrame, lag_order: int = 1) -> Prophet:
    """
    Fits an Autoregressive Prophet model to the provided DataFrame.
    ----------------

    Args:
        df: pandas DataFrame containing the time series
        lag_order: number of lagged values to use as regressors
    ----------------

    Returns:
        fitted Prophet model
    ----------------

    Mathematical explanation:
    Autoregressive Model:
    The autoregressive model in Prophet is designed to handle time series data that exhibit autoregressive behavior,
    meaning that the value of the time series at time t depends on its previous values.
    In this model, the trend component, g(t), is modeled using a piecewise linear model, the seasonality component, s(t), is modeled using a Fourier series,
    and the autoregressive component, ar(t), is modeled using an ARIMA model.
    The formula for the autoregressive model can be expressed as:

    y(t) = g(t) + s(t) + h(t) + ar(t) + e(t)

    where:
    y(t), g(t), s(t), h(t), and e(t) have the same meanings as in the additive model, and ar(t) represents the autoregressive component at time t.
    The autoregressive component, ar(t), is modeled using an ARIMA model, which is defined as:
    ar(t) = ∑(i=1)^p (φ(i) * y(t - i)) + ∑(j=1)^q (θ(j) * e(t - j))

    where:
        p represents the order of the autoregressive model
        q represents the order of the moving average model
        φ(i) represents the coefficient of the i-th lagged value of the time series
        θ(j) represents the coefficient of the j-th lagged value of the error term
        y(t - i) represents the value of the time series at time t - i
        e(t - j) represents the value of the error term at time t - j

    """
    # Create lagged values of the target variable
    for i in range(1, lag_order + 1):
        df[f'y_lag{i}'] = df['y'].shift(i)

    # Drop missing values
    df = df.dropna()

    # Fit Prophet model
    model = Prophet()
    for i in range(1, lag_order + 1):
        model.add_regressor(f'y_lag{i}')
    model.fit(df)
    return model


def forecast_ar_model(model: Prophet, periods: int, freq: str = 'D') -> pd.DataFrame:
    """
    Generates forecasts with an Autoregressive Prophet model.

    :param model: Prophet model object
    :param periods: number of periods to forecast
    :param freq: the frequency of the forecasted data
    :return: pandas DataFrame with forecast values and confidence intervals
    """
    future_df = model.make_future_dataframe(periods=periods, freq=freq)
    for i in range(1, model.extra_regressors_cols):
        future_df[f'y_lag{i}'] = future_df['y'].shift(i)
    forecast = model.predict(future_df)

    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]


def detect_prophet_outliers(time_series: pd.DataFrame, changepoint_prior_scale: float, sensitivity: float = 0.05, include: bool = True, **kwargs) -> tuple(pd.DataFrame, pd.DataFrame):
    """
        Detect outliers in a time series using Prophet.

        Args:
            time_series (pd.DataFrame): Time series to detect outliers in.
            changepoint_prior_scale (float): Parameter controlling the flexibility of the automatic changepoint selection.
            sensitivity (float): Parameter controlling the sensitivity of outlier detection.
                                 The higher the value, the more sensitive the algorithm is to outliers.
                                 Default is 0.1.
            include (bool): Whether to include detected outliers in the modeling process.
                             If True, outliers will be included, otherwise they will be excluded.
                             Default is True.

        Returns:
            outliers (pd.DataFrame): DataFrame containing the detected outliers.
            time_series (pd.DataFrame): DataFrame containing the time series with outliers removed.
            """
    time_series = time_series.reset_index().rename(columns={'index': 'ds', time_series.columns[0]: 'y'})

    model = Prophet(
        changepoint_prior_scale=changepoint_prior_scale, **kwargs)

    model.fit(time_series)
    forecast = model.predict(time_series)
    forecast['residuals'] = forecast['yhat'] - forecast['y']
    forecast['residuals_abs'] = abs(forecast['residuals'])
    forecast['residuals_abs_norm'] = forecast['residuals_abs'] / forecast['yhat_std']

    threshold = forecast['residuals_abs_norm'].quantile(1 - sensitivity)
    outliers = forecast[forecast['residuals_abs_norm'] > threshold]

    if not include:
        time_series = time_series[~time_series['ds'].isin(outliers['ds'])]

    return outliers, time_series