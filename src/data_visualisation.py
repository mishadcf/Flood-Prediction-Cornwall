import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

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

# Example usage:
# 
import os
import pandas as pd
import matplotlib.pyplot as plt

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
            


