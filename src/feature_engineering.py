
import pandas as pd
import numpy as np

def create_features(
    df: pd.DataFrame,
    target_column: str,
    weather_columns: list,
    max_lag: int = 6,
    rolling_windows: list = [3, 6, 12, 24],
    diff_lags: list = [1, 2],
    seasonal_period: int = 24
) -> pd.DataFrame:
    """
    Generates engineered features for time series forecasting.

    Parameters:
    - df (pd.DataFrame): Input DataFrame with datetime index.
    - target_column (str): Name of the target variable column.
    - weather_columns (list): List of weather variable column names.
    - max_lag (int): Maximum number of hours for lagged features.
    - rolling_windows (list): Window sizes (in hours) for rolling statistics and sums.
    - diff_lags (list): Lag intervals for differencing.
    - seasonal_period (int): Seasonal period for cyclical features.

    Returns:
    - pd.DataFrame: DataFrame with engineered features.
    """
    
    df = df.copy()
    
    ## 1. Target Variable Creation
    lead_time = 6  # Predicting 6 hours into the future
    df['target_t_plus_6'] = df[target_column].shift(-lead_time)
    
    ## 2. Lagged Features
    
    # a. Lagged River Level Features
    for lag in range(1, max_lag + 1):
        df[f'{target_column}_lag_{lag}h'] = df[target_column].shift(lag)
    
    # b. Lagged Weather Features
    for weather in weather_columns:
        for lag in range(0, max_lag + 1):  # Including current time (lag=0)
            df[f'{weather}_lag_{lag}h'] = df[weather].shift(lag)
    
    ## 3. Rolling Statistics
    
    # a. Rolling Means and Standard Deviations for River Level
    for window in rolling_windows:
        df[f'{target_column}_rolling_mean_{window}h'] = df[target_column].rolling(window=window).mean()
        df[f'{target_column}_rolling_std_{window}h'] = df[target_column].rolling(window=window).std()
    
    # b. Rolling Means and Standard Deviations for Rainfall
    for weather in weather_columns:
        if 'rain' in weather.lower():
            for window in rolling_windows:
                df[f'{weather}_rolling_sum_{window}h'] = df[weather].rolling(window=window).sum()
                df[f'{weather}_rolling_mean_{window}h'] = df[weather].rolling(window=window).mean()
                df[f'{weather}_rolling_std_{window}h'] = df[weather].rolling(window=window).std()
    
    ## 4. Differencing
    
    # a. Differencing for River Level
    for lag in diff_lags:
        df[f'{target_column}_diff_{lag}h'] = df[target_column].diff(lag)
    
    # b. Differencing for Rainfall  
    for weather in weather_columns:
        if 'rain' in weather.lower():
            for lag in diff_lags:
                df[f'{weather}_diff_{lag}h'] = df[weather].diff(lag)
    
    ## 5. Exponential Weighted Moving Averages (EWMA)
    
    # a. EWMA for River Level
    for span in rolling_windows:
        df[f'{target_column}_ewm_mean_{span}h'] = df[target_column].ewm(span=span, adjust=False).mean()
    
    # b. EWMA for Rainfall
    for weather in weather_columns:
        if 'rain' in weather.lower():
            for span in rolling_windows:
                df[f'{weather}_ewm_sum_{span}h'] = df[weather].ewm(span=span, adjust=False).mean() * span  # Approximation
    
    ## 6. Time-Based Features
    
    # a. Extracting Temporal Components
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek  # Monday=0, Sunday=6
    
    # b. Cyclical Encoding for Hour and Day of Week
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    ## 7. Interaction Features (Optional)
    
    # Example: Interaction between current rainfall and river level
    for weather in weather_columns:
        if 'rain' in weather.lower():
            df[f'{weather}_x_{target_column}'] = df[weather] * df[target_column]
    
    ## 8. Handling Missing Values
    
    # Dropping rows where target is NaN
    df.dropna(subset=['target_t_plus_6'], inplace=True)
    
    # Optionally, fill other NaN values
    # For simplicity, forward fill is used here
    df.fillna(method='ffill', inplace=True)
    
    # If any NaNs remain (e.g., at the very start), drop them
    df.dropna(inplace=True)
    
    ## 9. Feature Selection 
    
    return df
