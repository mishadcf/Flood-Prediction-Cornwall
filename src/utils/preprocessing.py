import pandas as pd
import numpy as np
from typing import Dict, Optional
from pandas.tseries.frequencies import to_offset

def process_river(df: pd.DataFrame, column: str = 'value') -> pd.DataFrame:
    """
    Core preprocessing for river gauge time series data.
    
    Steps:
    1. Replace negative values with NaN
    2. Remove outliers (values beyond mean ± 10*std)
    3. Resample to 15min frequency
    4. Seasonally impute missing values
    
    Args:
        df: DataFrame with datetime index and measurement column
        column: Name of the measurement column
        
    Returns:
        Processed DataFrame
    """
    # Copy to avoid modifying original
    df = df.copy()
    
    # Replace negative values with NaN
    df.loc[df[column] < 0, column] = np.nan
    
    # Remove outliers
    mean, std = df[column].mean(), df[column].std()
    df.loc[df[column] > mean + 10*std, column] = np.nan
    df.loc[df[column] < mean - 10*std, column] = np.nan
    
    # Resample to 15min frequency if needed
    if df.index.freq != '15min':
        df = df.resample('15min').asfreq()
    
    # Impute missing values
    df = seasonal_impute(df, column)
    
    return df

def seasonal_impute(df: pd.DataFrame, column: str = 'value') -> pd.DataFrame:
    """
    Impute missing values based on seasonal (monthly-hourly) averages.
    
    Args:
        df: DataFrame with datetime index
        column: Column name containing the values to impute
        
    Returns:
        DataFrame with missing values imputed
    """
    df = df.copy()
    
    # Extract time components
    month_hour = pd.DataFrame({
        'month': df.index.month,
        'hour': df.index.hour,
        'value': df[column]
    })
    
    # Calculate monthly-hourly averages
    seasonal_averages = month_hour.groupby(['month', 'hour'])['value'].mean()
    
    # Fill missing values with seasonal averages
    for idx in df[df[column].isna()].index:
        month, hour = idx.month, idx.hour
        if (month, hour) in seasonal_averages:
            df.loc[idx, column] = seasonal_averages[month, hour]
    
    return df

def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract time-based features from DataFrame index.
    
    Features:
    - Hour of day (cyclic)
    - Day of week (cyclic)
    - Month (cyclic)
    - Is weekend
    - Is holiday (UK holidays)
    
    Args:
        df: DataFrame with datetime index
        
    Returns:
        DataFrame with time features added
    """
    df = df.copy()
    
    # Cyclic encoding of time features
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    
    df['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df['day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    
    df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
    
    # Binary features
    df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)
    
    return df

def smart_linear_interpolate(df: pd.DataFrame, 
                           column: str = 'value', 
                           z_thresh: float = 10.0,
                           max_gap: str = '2H') -> pd.DataFrame:
    """
    Interpolate missing values only when neighboring values are reasonable.
    
    Args:
        df: DataFrame with datetime index
        column: Column to interpolate
        z_thresh: Z-score threshold for considering neighbors reasonable
        max_gap: Maximum time gap to interpolate across
        
    Returns:
        DataFrame with eligible missing values interpolated
    """
    df = df.copy()
    
    # Calculate statistics for reasonableness check
    mean = df[column].mean()
    std = df[column].std()
    lower_bound = mean - z_thresh * std
    upper_bound = mean + z_thresh * std
    
    # Find missing values
    missing_idx = df[df[column].isna()].index
    
    for idx in missing_idx:
        # Find nearest non-null values before and after
        left_val = df.loc[:idx, column].last_valid_index()
        right_val = df.loc[idx:, column].first_valid_index()
        
        if left_val is None or right_val is None:
            continue
            
        # Check if gap is too large
        if (right_val - left_val) > pd.Timedelta(max_gap):
            continue
            
        # Check if neighbors are reasonable
        left_value = df.loc[left_val, column]
        right_value = df.loc[right_val, column]
        
        if (lower_bound <= left_value <= upper_bound and 
            lower_bound <= right_value <= upper_bound):
            # Interpolate this specific gap
            temp_idx = pd.date_range(left_val, right_val, freq=df.index.freq)
            temp_series = pd.Series(index=temp_idx)
            temp_series.iloc[0] = left_value
            temp_series.iloc[-1] = right_value
            temp_series = temp_series.interpolate(method='linear')
            
            # Update the original dataframe
            df.loc[temp_idx, column] = temp_series
            
    return df
