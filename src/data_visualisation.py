import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from statsmodels.nonparametric.smoothers_lowess import lowess

if __name__ == '__main__':
    pass

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
    plt.plot(df['time'], df[variable], label=variable)
    plt.title(f"Time Series of {variable}" + (f" at {station_name}" if station_name else ""))
    plt.xlabel("time")
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

def plot_correlation_matrix(df, save_path=None):
    """
    Plot correlation matrix for a DataFrame.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def visualize_missing_values_rivers(csv_path: str, value_column: str = 'value'):
    """
    Visualizes the missing data patterns by hour, day of the week, and month for a time-indexed CSV file.

    Parameters:
    - csv_path: str - The path to the CSV file containing river gauge data.
    - value_column: str - The column name for the data values to check for missing values (default is 'value').
    """
    # Load data
    try:
        df = pd.read_csv(csv_path, parse_dates=['time'], index_col='time')
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
    
    # Ensure that 'value' column exists
    if value_column not in df.columns:
        print(f"Error: Column '{value_column}' not found in the CSV.")
        return
    
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



def visualize_missing_patterns_bulk(directory: str, output_dir: str, value_column: str = 'value'):
    """
    Generates and saves missing data pattern visualizations for all CSV files in a directory after resampling to 15-minute intervals.

    Parameters:
    - directory: str - The directory containing the CSV files.
    - output_dir: str - The directory where plots will be saved.
    - value_column: str - The column name for the data values to check for missing values (default is 'value').
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Loop through all CSV files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            csv_path = os.path.join(directory, filename)
            
            # Load data
            try:
                df = pd.read_csv(csv_path, parse_dates=['time'], index_col='time')
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                continue
            
            # Ensure 'value' column exists
            if value_column not in df.columns:
                print(f"Column '{value_column}' not found in {filename}")
                continue

            # Resample to 15-minute frequency and fill missing timestamps with NaN
            df_resampled = df.asfreq('15min')
            
            # Check for missing values in the resampled data
            missing_count = df_resampled[value_column].isnull().sum()
            if missing_count == 0:
                print(f"No missing values found in resampled data for {filename}. Skipping...")
                continue
            
            print(f"Processing {filename} with {missing_count} missing values after resampling.")

            # Create missing value indicator
            df_resampled['missing'] = df_resampled[value_column].isnull().astype(int)
            
            # Generate and save plots for missing values by hour, day, and month
            plt.figure(figsize=(10, 5))
            missing_by_hour = df_resampled.groupby(df_resampled.index.hour)['missing'].mean() * 100
            missing_by_hour.plot(kind='bar', color='skyblue', edgecolor='black')
            plt.title(f"Percentage of Missing Values by Hour of Day - {filename}")
            plt.xlabel("Hour of Day")
            plt.ylabel("Percentage of Missing Values")
            plt.savefig(os.path.join(output_dir, f"{filename}_missing_by_hour.png"))
            plt.close()

            plt.figure(figsize=(10, 5))
            missing_by_day = df_resampled.groupby(df_resampled.index.dayofweek)['missing'].mean() * 100
            missing_by_day.plot(kind='bar', color='salmon', edgecolor='black')
            plt.title(f"Percentage of Missing Values by Day of Week - {filename}")
            plt.xlabel("Day of Week (0=Monday, 6=Sunday)")
            plt.ylabel("Percentage of Missing Values")
            plt.savefig(os.path.join(output_dir, f"{filename}_missing_by_day.png"))
            plt.close()

            plt.figure(figsize=(10, 5))
            missing_by_month = df_resampled.groupby(df_resampled.index.month)['missing'].mean() * 100
            missing_by_month.plot(kind='bar', color='lightgreen', edgecolor='black')
            plt.title(f"Percentage of Missing Values by Month - {filename}")
            plt.xlabel("Month")
            plt.ylabel("Percentage of Missing Values")
            plt.savefig(os.path.join(output_dir, f"{filename}_missing_by_month.png"))
            plt.close()

            print(f"Plots saved for {filename}")
            
def analyze_seasonal_missingness(directory, value_column='value'):
    monthly_missing = []
    
    for filename in os.listdir(directory):
        if filename.endswith('_raw.csv'):
            # Load and resample data
            df = pd.read_csv(os.path.join(directory, filename), parse_dates=['time'], index_col='time')
            df_resampled = df.asfreq('15min')
            df_resampled['missing'] = df_resampled[value_column].isnull().astype(int)

            # Calculate monthly missingness percentage
            monthly_missing_df = df_resampled['missing'].groupby(df_resampled.index.month).mean() * 100
            monthly_missing_df = monthly_missing_df.rename(filename)
            monthly_missing.append(monthly_missing_df)
    
    # Concatenate monthly missing percentages across all files
    monthly_missing_combined = pd.concat(monthly_missing, axis=1)
    monthly_mean_missing = monthly_missing_combined.mean(axis=1)

    # Plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=monthly_mean_missing, marker='o', color='b')
    plt.title('Average Percentage of Missing Values by Month')
    plt.xlabel('Month')
    plt.ylabel('Average Percentage of Missing Values')
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.show()


def save_loess_plots(directory, output_dir, frac=0.1, max_plots=None):
    """
    Apply LOESS smoothing to each river gauge CSV file in a directory and save the plots.

    Parameters:
    - directory: str, path to the directory containing river gauge CSV files.
    - output_dir: str, path to the directory where the plots will be saved.
    - frac: float, the fraction of data to use for LOESS smoothing. Default is 0.1.
    - max_plots: int or None, maximum number of plots to generate. Set to None for all files.
    """
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    plot_count = 0
    for filename in os.listdir(directory):
        if filename.endswith('_raw.csv'):
            file_path = os.path.join(directory, filename)
            
            # Load the data
            df = pd.read_csv(file_path, parse_dates=['time'], index_col='time')
            
            # Ensure data is resampled to fill missing timestamps at 15min intervals
            df = df.asfreq('15min')
            
            # Check if 'value' column exists
            if 'value' not in df.columns:
                print(f"Column 'value' not found in {filename}")
                continue
            
            # Apply LOESS smoothing to the 'value' column
            df['value'] = df['value'].interpolate(method='linear')  # Interpolate to handle NaNs
            loess_smoothed = lowess(df['value'].values, np.arange(len(df)), frac=frac, return_sorted=False)

            # Plot and save as a .png file
            plt.figure(figsize=(12, 6))
            plt.plot(df.index, df['value'], label='Original Data', color='blue', alpha=0.6)
            plt.plot(df.index, loess_smoothed, label='LOESS Smoothed', color='red', linewidth=2)
            plt.title(f"LOESS Smoothing for {filename}")
            plt.xlabel("Time")
            plt.ylabel("River Gauge Value")
            plt.legend()
            
            # Save the plot
            output_path = os.path.join(output_dir, f"{filename}_loess_plot.png")
            plt.savefig(output_path)
            plt.close()  # Close the plot to free memory
            plot_count += 1
            print(f"Plot saved for {filename} at {output_path}")
            
            # Limit the number of plots (optional)
            if max_plots and plot_count >= max_plots:
                print(f"Stopped after generating {max_plots} plots.")
                break

  
def process_and_plot_csvs(cleaned_dir="data/river_data/highest_granularity/cleaned"):
    """
    The function `process_and_plot_csvs` reads cleaned CSV files, identifies outliers and negative
    values, and plots the data with outliers highlighted in red and threshold lines.

    :param cleaned_dir: The `cleaned_dir` parameter in the `process_and_plot_csvs` function is the
    directory path where the cleaned CSV files are located. This function reads each CSV file in the
    specified directory, processes the data, identifies outliers, checks for negative values, and then
    plots the data along with outliers, defaults to data/river_data/highest_granularity/cleaned
    (optional)
    """

    # Create a list of all CSV files in the directory
    csv_files = [f for f in os.listdir(cleaned_dir) if f.endswith(".csv")]

    for csv_file in csv_files:
        file_path = os.path.join(cleaned_dir, csv_file)
        print(f"\nProcessing: {file_path}")

        # Read the cleaned CSV
        df = pd.read_csv(file_path, parse_dates=['time'], index_col='time')

        # Check that 'value' column exists
        if 'value' not in df.columns:
            print(f"Warning: 'value' column not found in {csv_file}. Skipping...")
            continue

        # Compute mean and std
        mean_val = df['value'].mean()
        std_val = df['value'].std()

        # Calculate thresholds for outliers
        threshold_upper = mean_val + 10 * std_val
        threshold_lower = mean_val - 10 * std_val

        # Identify outliers
        outliers = df[(df['value'] > threshold_upper) | (df['value'] < threshold_lower)]
        num_outliers = len(outliers)

        # Check for negative values
        negative_values = df[df['value'] < 0]
        num_negative = len(negative_values)

        # Print summary
        print(f"Mean: {mean_val:.2f}, Std: {std_val:.2f}")
        print(f"Thresholds: Lower = {threshold_lower:.2f}, Upper = {threshold_upper:.2f}")
        print(f"Outliers beyond thresholds: {num_outliers}")
        if num_outliers > 0:
            print("Outlier timestamps and values:")
            print(outliers[['value']].head(10))  # print first 10 outliers for inspection

        if num_negative > 0:
            print(f"Warning: {num_negative} values are negative. Example:")
            print(negative_values[['value']].head(10))
        else:
            print("No negative values found.")

        # Plot the data
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['value'], label='Value', color='blue')

        # Highlight outliers in red
        if num_outliers > 0:
            plt.scatter(outliers.index, outliers['value'], color='red', label='Outliers')

        # Add horizontal lines at thresholds
        plt.axhline(y=threshold_upper, color='orange', linestyle='--', label='Upper Threshold')
        plt.axhline(y=threshold_lower, color='green', linestyle='--', label='Lower Threshold')

        plt.title(f"{csv_file}: Value over Time")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()

        # Uncomment to save plots instead of showing
        # plot_filename = f"{csv_file.replace('.csv', '')}_check.png"
        # plt.savefig(os.path.join("check_plots", plot_filename))

        plt.show()
        plt.close()            
        
        
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import skew, zscore

def river_gauge_eda(df, 
                    time_col='datetime', 
                    level_col='water_level', 
                    anomaly_method='zscore', 
                    z_thresh=3,
                    seasonality_period='year'):
    """
    Perform EDA on a single gauge's time series data with fixed 15-minute frequency
    and annual seasonal decomposition.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing time series data with at least two columns:
        1) A datetime column (time_col).
        2) A water level column (level_col).
        
    time_col : str
        Name of the datetime column in df.
    
    level_col : str
        Name of the gauge level column in df.
    
    anomaly_method : str
        Method to detect anomalies. Options: 'zscore' or 'iqr'.
    
    z_thresh : float
        Threshold for z-score based outlier detection. E.g., 3 means detect points > 3 std dev away.
    
    seasonality_period : str
        The seasonal period for decomposition. Options:
            - 'year': Annual seasonality
            - 'week': Weekly seasonality
            - 'day': Daily seasonality
        Default is 'year'.
    
    Returns
    -------
    dict
        A dictionary of results containing:
          - 'summary_stats': basic stats of the water level
          - 'missing_values': total missing values
          - 'negative_values': total negative values (if any)
          - 'skewness': skew of the distribution
          - 'outliers': index of detected outliers
          - 'seasonality_plots': decomposition plots (if successful)
          - 'figures': dictionary of figures/axes references for further usage
    """
    
    # ---- 1. Ensure datetime format and sort ---------------------------------
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')  # Force datetime
    df.sort_values(by=time_col, inplace=True)
    df.set_index(time_col, inplace=True)
    
    # ---- 2. Align to 15-Minute Frequency and Detect Missing Intervals -------
    freq = '15T'  # Fixed to 15-minute frequency
    df_aligned = df.asfreq(freq)
    total_missing = df_aligned[level_col].isnull().sum()
    
    # ---- 3. Handle Duplicates by Resampling with Mean ----------------------
    # Resampling ensures that any duplicate entries within the same 15-minute interval are averaged
    df_resampled = df_aligned.resample(freq).mean()
    
    # ---- 4. Basic Summary Statistics ----------------------------------------
    summary_stats = df_resampled[level_col].describe()
    total_negative = (df_resampled[level_col] < 0).sum()
    data_skewness = skew(df_resampled[level_col].dropna())
    
    # ---- 5. Plot the Raw Time Series ---------------------------------------
    fig_time, ax_time = plt.subplots(figsize=(15, 5))
    ax_time.plot(df_resampled.index, df_resampled[level_col], label='River Gauge Level')
    ax_time.set_title("Time Series of River Gauge Level", fontsize=14)
    ax_time.set_xlabel("Time", fontsize=12)
    ax_time.set_ylabel("Water Level", fontsize=12)
    ax_time.legend()
    plt.tight_layout()
    
    # ---- 6. Distribution and Boxplot ---------------------------------------
    fig_dist, ax_dist = plt.subplots(1, 2, figsize=(15, 5))
    
    # Histogram with KDE
    sns.histplot(df_resampled[level_col], kde=True, ax=ax_dist[0], color='skyblue')
    ax_dist[0].set_title("Distribution of Water Levels", fontsize=14)
    ax_dist[0].set_xlabel("Water Level", fontsize=12)
    
    # Boxplot
    sns.boxplot(y=df_resampled[level_col], ax=ax_dist[1], color='lightgreen')
    ax_dist[1].set_title("Boxplot of Water Levels", fontsize=14)
    ax_dist[1].set_ylabel("Water Level", fontsize=12)
    
    plt.tight_layout()
    
    # ---- 7. Detect Outliers ------------------------------------------------
    outliers = []
    
    if anomaly_method == 'zscore':
        # Z-score method
        zscores = np.abs(zscore(df_resampled[level_col].dropna()))
        outlier_indices = df_resampled[level_col].dropna().index[zscores > z_thresh]
        outliers = outlier_indices.tolist()
    
    elif anomaly_method == 'iqr':
        # IQR method
        Q1 = df_resampled[level_col].quantile(0.25)
        Q3 = df_resampled[level_col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outlier_indices = df_resampled[(df_resampled[level_col] < lower_bound) | 
                                       (df_resampled[level_col] > upper_bound)].index
        outliers = outlier_indices.tolist()
    
    # ---- 8. Seasonal Decomposition -----------------------------------------
    decomposition_plots = None
    try:
        # Fill missing data via interpolation
        df_decomp = df_resampled[level_col].interpolate(method='time').ffill().bfill()
        
        # Define seasonal period based on desired seasonality
        if seasonality_period == 'year':
            # Number of 15-minute intervals in a year
            period = 96 * 365  # 96 intervals/day * 365 days
        elif seasonality_period == 'week':
            period = 96 * 7  # Weekly seasonality
        elif seasonality_period == 'day':
            period = 96  # Daily seasonality
        else:
            raise ValueError("Invalid 'seasonality_period'. Choose from 'year', 'week', or 'day'.")
        
        # Perform seasonal decomposition
        decomposition = seasonal_decompose(df_decomp, model='additive', period=period)
        
        # Plot decomposition
        fig_decomp = decomposition.plot()
        fig_decomp.set_size_inches(15, 10)
        fig_decomp.suptitle("Seasonal Decomposition of River Gauge Levels", fontsize=16)
        plt.tight_layout()
        
        decomposition_plots = fig_decomp
    
    except Exception as e:
        print(f"Seasonal decomposition failed: {e}")
    
    # ---- 9. Return Results -------------------------------------------------
    results = {
        'summary_stats': summary_stats,
        'missing_values': total_missing,
        'negative_values': total_negative,
        'skewness': data_skewness,
        'outliers': outliers,
        'seasonality_plots': decomposition_plots,
        'figures': {
            'time_series_plot': (fig_time, ax_time),
            'distribution_plots': (fig_dist, ax_dist)
        }
    }
    
    return results



