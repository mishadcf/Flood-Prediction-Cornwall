import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.tsa.seasonal import seasonal_decompose
import os

def plot_time_series(df, variable, station_name=None, save_path=None):
    """
    Plot time series data for a given variable.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing time series data.
        variable (str): Column name of the variable to plot.
        station_name (str): Name of the river gauge or weather station (optional).
        save_path (str): File path to save the plot, if desired.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df[variable], label=variable)
    plt.title(f"Time Series of {variable}" + (f" at {station_name}" if station_name else ""))
    plt.xlabel("Time")
    plt.ylabel(variable)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_histogram(df, variable, station_name=None, save_path=None):
    """
    Plot a histogram for a given variable.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df[variable], kde=True)
    plt.title(f"Distribution of {variable}" + (f" at {station_name}" if station_name else ""))
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def visualize_missing_values_rivers(df: pd.DataFrame, value_column: str = 'value'):
    """
    Visualizes the missing data patterns by hour, day of the week, and month.

    Parameters:
    - df: DataFrame with datetime index and value column
    - value_column: The column name for the data values to check for missing values
    """
    # Create a binary column indicating missing values
    df['missing'] = df[value_column].isnull().astype(int)

    # Visualization of missing values by hour of the day
    missing_by_hour = df.groupby(df.index.hour)['missing'].mean() * 100
    plt.figure(figsize=(10, 5))
    missing_by_hour.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title("Percentage of Missing Values by Hour of Day")
    plt.xlabel("Hour of Day")
    plt.ylabel("Percentage of Missing Values")
    plt.show()

    # Visualization of missing values by day of the week
    missing_by_day = df.groupby(df.index.dayofweek)['missing'].mean() * 100
    plt.figure(figsize=(10, 5))
    missing_by_day.plot(kind='bar', color='salmon', edgecolor='black')
    plt.title("Percentage of Missing Values by Day of Week")
    plt.xlabel("Day of Week (0=Monday, 6=Sunday)")
    plt.ylabel("Percentage of Missing Values")
    plt.show()

    # Visualization of missing values by month
    missing_by_month = df.groupby(df.index.month)['missing'].mean() * 100
    plt.figure(figsize=(10, 5))
    missing_by_month.plot(kind='bar', color='lightgreen', edgecolor='black')
    plt.title("Percentage of Missing Values by Month")
    plt.xlabel("Month")
    plt.ylabel("Percentage of Missing Values")
    plt.show()

def river_gauge_eda(df, 
                   time_col='datetime', 
                   level_col='value', 
                   anomaly_method='zscore', 
                   z_thresh=3,
                   seasonality_period='year'):
    """
    Perform EDA on a single gauge's time series data.
    
    Parameters:
        df: DataFrame with datetime index and value column
        time_col: Name of datetime column
        level_col: Name of water level column
        anomaly_method: Method to detect anomalies ('zscore' or 'iqr')
        z_thresh: Threshold for z-score based outlier detection
        seasonality_period: Period for seasonal decomposition
    
    Returns:
        dict: Dictionary containing analysis results and figures
    """
    results = {}
    figures = {}

    # Basic statistics
    results['summary_stats'] = df[level_col].describe()
    results['missing_values'] = df[level_col].isna().sum()
    results['negative_values'] = (df[level_col] < 0).sum()
    results['skewness'] = df[level_col].skew()

    # Time series plot
    fig, ax = plt.subplots(figsize=(15, 5))
    df[level_col].plot(ax=ax)
    ax.set_title('Water Level Time Series')
    figures['time_series'] = fig

    # Distribution plot
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df[level_col].dropna(), kde=True, ax=ax)
    ax.set_title('Water Level Distribution')
    figures['distribution'] = fig

    # Detect outliers
    if anomaly_method == 'zscore':
        z_scores = np.abs((df[level_col] - df[level_col].mean()) / df[level_col].std())
        outliers = z_scores > z_thresh
    else:  # IQR method
        Q1 = df[level_col].quantile(0.25)
        Q3 = df[level_col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = (df[level_col] < (Q1 - 1.5 * IQR)) | (df[level_col] > (Q3 + 1.5 * IQR))
    
    results['outliers'] = df[outliers].index.tolist()

    # Plot with outliers highlighted
    fig, ax = plt.subplots(figsize=(15, 5))
    df[level_col].plot(ax=ax, alpha=0.7, label='Normal')
    df[outliers][level_col].plot(ax=ax, style='r.', label='Outliers', alpha=0.5)
    ax.set_title('Water Level Time Series with Outliers')
    ax.legend()
    figures['outliers'] = fig

    # Seasonal decomposition
    try:
        if seasonality_period == 'year':
            period = 365 * 24  # Assuming hourly data
        elif seasonality_period == 'week':
            period = 7 * 24
        else:  # daily
            period = 24

        decomposition = seasonal_decompose(
            df[level_col].dropna(),
            period=period,
            extrapolate_trend=True
        )

        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        decomposition.observed.plot(ax=axes[0])
        axes[0].set_title('Observed')
        decomposition.trend.plot(ax=axes[1])
        axes[1].set_title('Trend')
        decomposition.seasonal.plot(ax=axes[2])
        axes[2].set_title('Seasonal')
        decomposition.resid.plot(ax=axes[3])
        axes[3].set_title('Residual')
        plt.tight_layout()
        figures['decomposition'] = fig
        
    except Exception as e:
        print(f"Could not perform seasonal decomposition: {e}")

    results['figures'] = figures
    return results

def save_loess_plots(df, output_dir, frac=0.1):
    """
    Apply LOESS smoothing to time series data and save the plots.
    
    Parameters:
        df: DataFrame with datetime index and value column
        output_dir: Directory to save plots
        frac: Fraction of data to use for LOESS smoothing
    """
    os.makedirs(output_dir, exist_ok=True)
    
    y = df['value'].values
    x = np.arange(len(y))
    
    # Apply LOESS smoothing
    z = lowess(y, x, frac=frac)
    
    # Plot and save
    plt.figure(figsize=(15, 5))
    plt.plot(df.index, y, label='Original', alpha=0.5)
    plt.plot(df.index, z[:, 1], 'r-', label='LOESS Trend', linewidth=2)
    plt.title('Time Series with LOESS Trend')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'loess_trend.png'))
    plt.close()

def plot_model_evaluation(actual, predicted, title=None):
    """
    Create evaluation plots for model predictions.
    
    Parameters:
        actual: Series of actual values
        predicted: Series of predicted values
        title: Optional title prefix for plots
    """
    title_prefix = f"{title} - " if title else ""
    
    # Actual vs Predicted
    plt.figure(figsize=(15, 5))
    plt.plot(actual.index, actual, label='Actual', alpha=0.7)
    plt.plot(predicted.index, predicted, label='Predicted', alpha=0.7)
    plt.title(f"{title_prefix}Actual vs Predicted Values")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Residuals over time
    residuals = actual - predicted
    plt.figure(figsize=(15, 5))
    plt.plot(actual.index, residuals)
    plt.title(f"{title_prefix}Prediction Residuals")
    plt.grid(True)
    plt.show()
    
    # Residual distribution
    plt.figure(figsize=(10, 5))
    sns.histplot(residuals, kde=True)
    plt.title(f"{title_prefix}Residual Distribution")
    plt.show()
    
    # Scatter plot of actual vs predicted
    plt.figure(figsize=(10, 10))
    plt.scatter(actual, predicted, alpha=0.5)
    plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--')
    plt.title(f"{title_prefix}Actual vs Predicted Scatter")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.grid(True)
    plt.show()
