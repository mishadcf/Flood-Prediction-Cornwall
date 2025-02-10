import pandas as pd
import numpy as np
from typing import List, Optional

def create_features(
    df: pd.DataFrame,
    target_column: str,
    weather_columns: Optional[List[str]] = None,
    max_lag: int = 6,
    rolling_windows: List[int] = [3, 6, 12, 24],
    diff_lags: List[int] = [1, 2],
    seasonal_period: int = 24
) -> pd.DataFrame:
    """
    Generates engineered features for time series forecasting.

    Parameters:
    - df (pd.DataFrame): Input DataFrame with datetime index
    - target_column (str): Name of the target variable column
    - weather_columns (list): Optional list of weather variable column names
    - max_lag (int): Maximum number of hours for lagged features
    - rolling_windows (list): Window sizes (in hours) for rolling statistics
    - diff_lags (list): Lag intervals for differencing
    - seasonal_period (int): Seasonal period for cyclical features

    Returns:
    - pd.DataFrame: DataFrame with engineered features
    """
    df = df.copy()
    weather_columns = weather_columns or []
    
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
                df[f'{weather}_ewm_sum_{span}h'] = df[weather].ewm(span=span, adjust=False).mean() * span
    
    ## 6. Time-Based Features
    
    # a. Extracting Temporal Components
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek  # Monday=0, Sunday=6
    df['month'] = df.index.month
    
    # b. Cyclical Encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    ## 7. Interaction Features
    
    # Example: Interaction between current rainfall and river level
    for weather in weather_columns:
        if 'rain' in weather.lower():
            df[f'{weather}_x_{target_column}'] = df[weather] * df[target_column]
    
    ## 8. Handle Missing Values
    
    # Drop rows where target is NaN
    df.dropna(subset=['target_t_plus_6'], inplace=True)
    
    # Forward fill remaining NaN values
    df.fillna(method='ffill', inplace=True)
    
    # If any NaNs remain at the start, drop them
    df.dropna(inplace=True)
    
    return df

def create_sequence_features(
    df: pd.DataFrame,
    target_column: str,
    sequence_length: int = 24,
    step_size: int = 1,
    weather_columns: Optional[List[str]] = None
) -> tuple:
    """
    Create sequence features for deep learning models (LSTM, Transformer).
    
    Parameters:
    - df: DataFrame with datetime index
    - target_column: Name of target variable column
    - sequence_length: Length of input sequences
    - step_size: Number of steps between sequences
    - weather_columns: Optional list of weather feature columns
    
    Returns:
    - tuple: (X sequences, y targets)
    """
    feature_columns = [target_column]
    if weather_columns:
        feature_columns.extend(weather_columns)
    
    X, y = [], []
    
    for i in range(0, len(df) - sequence_length - 1, step_size):
        X.append(df[feature_columns].iloc[i:i + sequence_length].values)
        y.append(df[target_column].iloc[i + sequence_length])
    
    return np.array(X), np.array(y)

def select_features(
    df: pd.DataFrame,
    target_column: str,
    correlation_threshold: float = 0.1,
    max_features: Optional[int] = None
) -> List[str]:
    """
    Select most relevant features based on correlation with target.
    
    Parameters:
    - df: DataFrame with features
    - target_column: Name of target variable
    - correlation_threshold: Minimum absolute correlation to keep feature
    - max_features: Maximum number of features to select
    
    Returns:
    - list: Selected feature names
    """
    # Calculate correlations with target
    correlations = df.corr()[target_column].abs()
    
    # Filter features based on correlation threshold
    selected_features = correlations[correlations >= correlation_threshold].index.tolist()
    
    # Remove target from features list
    if target_column in selected_features:
        selected_features.remove(target_column)
    
    # Limit number of features if specified
    if max_features and len(selected_features) > max_features:
        selected_features = (correlations[selected_features]
                           .sort_values(ascending=False)
                           .head(max_features)
                           .index.tolist())
    
    return selected_features
