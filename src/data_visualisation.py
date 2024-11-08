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
    plt.plot(df['date'], df[variable], label=variable)
    plt.title(f"Time Series of {variable}" + (f" at {station_name}" if station_name else ""))
    plt.xlabel("Date")
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


