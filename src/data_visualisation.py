import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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
# visualize_missing_patterns('data/river_data/your_river_gauge_file.csv')


