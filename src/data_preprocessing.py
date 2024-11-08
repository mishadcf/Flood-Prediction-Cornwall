import pandas as pd
from ast import literal_eval
import json
import os
import numpy as np
from datetime import datetime
from pandas.tseries.frequencies import to_offset


"""utility functions for data preprocessing """


def extract_time_values_from_csv(path: str = None) -> pd.DataFrame:
    """extracts just the measurements, as per the format of the API response"""

    df = pd.read_csv(path, usecols=["values"])
    # Convert string representation of lists into actual lists of dictionaries

    df["values"] = df["values"].apply(literal_eval)

    df = df.explode("values")

    if not isinstance(df.at[0, "values"], list):
        df["values"] = df["values"].apply(lambda x: [x])  # Ensure it's a list
        df = df.explode("values")  # Now explode the DataFrame

        # Convert dictionaries to separate columns
        df = pd.concat(
            [df.drop("values", axis=1), df["values"].apply(pd.Series)], axis=1
        )

        # Convert 'time' to datetime format
        df["time"] = pd.to_datetime(df["time"])

        df = df.set_index("time")

        return df


def concat_all_river_gauges(river_directory="get_river_data/data"):
    all_dfs = []

    # Iterate through each file in the directory
    for item in os.listdir(river_directory):
        # Construct full file path
        file_path = os.path.join(river_directory, item)

        # Check if the item is a file and ends with '.csv'
        if os.path.isfile(file_path) and item.endswith(".csv"):
            # Read the CSV file
            df = pd.read_csv(file_path, index_col=0)

            # Extract gauge name from the file name (assuming the file name is the gauge name)
            gauge_name = os.path.splitext(item)[0]

            # Add a new level to the index with the gauge name
            df["gauge"] = gauge_name
            df.set_index("gauge", append=True, inplace=True)

            # Append the DataFrame to the list
            all_dfs.append(df)

    # Concatenate all DataFrames in the list with a multi-level index
    if all_dfs:
        result_df = pd.concat(all_dfs, axis=0)
    else:
        result_df = pd.DataFrame()  # Return an empty DataFrame if no CSV files found

    return result_df



def check_missing_days_in_csv(file_path):
    """Check a single CSV for missing days in the river gauge data and calculate the percentage of missing days."""
    # Read the CSV file
    river_data = pd.read_csv(file_path)
    
    # Convert the 'time' column to datetime (adjust column name if needed)
    river_data['time'] = pd.to_datetime(river_data['time'])
    
    # Resample the data to daily frequency
    river_data_daily = river_data.set_index('time').resample('D').asfreq()
    
    # Find missing days
    missing_days = river_data_daily[river_data_daily.isnull().any(axis=1)].index
    
    # Calculate total number of days and percentage of missing days
    total_days = len(river_data_daily)
    missing_percentage = (len(missing_days) / total_days) * 100 if total_days > 0 else 0
    
    return missing_days, missing_percentage

def check_missing_days_in_directory(directory):
    """Check all CSV files in a directory for missing days and calculate their percentages."""
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    files_with_missing_days = {}
    
    for file_name in csv_files:
        file_path = os.path.join(directory, file_name)
        missing_days, missing_percentage = check_missing_days_in_csv(file_path)
        
        if len(missing_days) > 0:
            files_with_missing_days[file_name] = {
                'num_missing_days': len(missing_days),
                'missing_percentage': missing_percentage
            }
    
    return files_with_missing_days

def remove_negative_river_levels(df):
    df['value'] =  df['value'].apply(lambda x : np.nan if x<0 else x)
    return df

def count_missing_quarter_hour_rows(df, filename):
    # Ensure the timestamp column is in datetime format and set as index
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])  
        df.set_index('time', inplace=True)
    else:
        return {"filename": filename, "total_missing_rows": None, "pct_missing": None, "error": "No 'time' column"}

    # Calculate the expected total number of 15-minute intervals
    start_time = df.index.min()
    end_time = df.index.max()
    expected_total_rows = pd.date_range(start=start_time, end=end_time, freq='15min').shape[0]
    
    # Resample to 15-minute intervals and count missing rows
    resampled = df.asfreq('15min')
    total_missing_rows = resampled.isnull().sum().sum()  # Total missing values across all columns
    pct_missing = (total_missing_rows / expected_total_rows) * 100.0

    # Return results as a dictionary
    return {"filename": filename, "total_missing_rows": total_missing_rows, "pct_missing": pct_missing}





def detect_frequency(df: pd.DataFrame) -> str:
    """Detect the most common time interval between observations in a time series DataFrame."""
    time_diffs = df.index.to_series().diff().dropna()
    most_common_diff = time_diffs.mode()[0]
    detected_frequency = to_offset(most_common_diff).freqstr
    return detected_frequency

def detect_frequency_in_directory(directory_path: str, output_path: str):
    # Initialize a list to store the results
    results = []
    errors = []

    # Loop through each CSV file in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory_path, filename)
            try:
                df = pd.read_csv(file_path)

                # Check if 'time' column exists
                if 'time' not in df.columns:
                    errors.append({'filename': filename, 'error': 'Missing "time" column'})
                    print(f"Error: '{filename}' is missing the 'time' column.")
                    continue  # Skip this file

                # Ensure 'time' column is in datetime format and set it as the index
                df['time'] = pd.to_datetime(df['time'])
                df.set_index('time', inplace=True)
                
                # Detect frequency
                detected_frequency = detect_frequency(df)
                print(f"Detected frequency for {filename}: {detected_frequency}")
                
                # Append results
                results.append({
                    'filename': filename,
                    'detected_frequency': detected_frequency
                })

            except Exception as e:
                # Log other errors
                errors.append({'filename': filename, 'error': str(e)})
                print(f"Error processing {filename}: {e}")

    # Convert results to a DataFrame and save as CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"Frequency information saved to {output_path}")

    # Save errors to a separate file if any
    if errors:
        errors_df = pd.DataFrame(errors)
        errors_output_path = 'detection_errors.csv'
        errors_df.to_csv(errors_output_path, index=False)
        print(f"Errors saved to {errors_output_path}")



def clean_river_csv(path: str, downsample_to_hourly=False, aggregation_method='mean', calculate_missing_measurements=True) -> pd.DataFrame:
    print(f"\nProcessing file: {path}")
    
    # Read CSV file
    try:
        original_df = pd.read_csv(path)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None, None

    # Check if 'time' column exists
    if 'time' not in original_df.columns:
        print(f"Error: 'time' column is missing in {path}")
        return original_df, None

    # Initial row count before parsing
    initial_row_count = len(original_df)
    
    # Convert 'time' column to datetime format with error coercion, then drop NaT rows
    original_df['time'] = pd.to_datetime(original_df['time'], errors='coerce')
    parsed_row_count = original_df['time'].notna().sum()  # Count rows with valid datetime values

    # Calculate percentage of rows dropped
    rows_dropped = initial_row_count - parsed_row_count
    pct_rows_dropped = (rows_dropped / initial_row_count) * 100
    print(f"Rows dropped due to unparseable 'time' values: {rows_dropped} ({pct_rows_dropped:.2f}%)")

    # Drop rows with NaT in 'time' column and set it as index
    original_df = original_df.dropna(subset=['time'])
    original_df.set_index('time', inplace=True)
    
    # Make into UTC timezone, for ease of merging on time series weather data
    original_df.index = original_df.index.tz_localize('UTC')
    
    # Resample the DataFrame to a 15-minute frequency for filling missing data
    df_15min = original_df.asfreq('15min')
    print(f"The resampling to 15 created {len(df_15min) - len(original_df)} rows corresponding to missing timestamps")

    # Optional calculation of missing rows information
    if calculate_missing_measurements:
        # Calculate the expected number of rows and the percentage of missing rows
        start_time = df_15min.index.min()
        end_time = df_15min.index.max()
        expected_total_rows = pd.date_range(start=start_time, end=end_time, freq='15min').shape[0]
        total_missing_rows = df_15min.isnull().sum().sum()  # Total missing values across all columns
        pct_missing = (total_missing_rows / expected_total_rows) * 100.0

        print(f"Total missing 15-minute rows: {total_missing_rows} ({pct_missing:.2f}%)\n")

    # Replace negative values with NaN and interpolate missing values
    df_15min['value'] = df_15min['value'].apply(lambda x: np.nan if x < 0 else x)
    df_15min['value'] = df_15min['value'].interpolate(method='linear')
    print("Negative values replaced and missing values filled using linear interpolation.\n")
    
    # Optional downsampling to hourly frequency
    if downsample_to_hourly:
        if aggregation_method == 'mean':
            df_hourly = df_15min.resample('H').mean()
        elif aggregation_method == 'first':
            df_hourly = df_15min.resample('H').first()
        else:
            raise ValueError("Invalid aggregation_method. Use 'mean' or 'first'.")
        
        print("Downsampling to hourly frequency with chosen aggregation method.\n")
        return original_df, df_hourly
    
    return original_df, df_15min


def extract_time_features(df):
    """
    Extracts time-related features from the index of the input DataFrame
    and returns a new DataFrame containing these features.

    Parameters:
    df (pd.DataFrame): DataFrame with a datetime index.

    Returns:
    pd.DataFrame: DataFrame containing the extracted time features.
    """
    # Ensure the DataFrame index is a datetime type
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("The DataFrame index must be a DatetimeIndex.")

    # Create a new DataFrame for time features
    time_features_df = pd.DataFrame(index=df.index)
    time_features_df['year'] = df.index.year
    time_features_df['month'] = df.index.month
    time_features_df['week'] = df.index.isocalendar().week
    time_features_df['hour'] = df.index.hour
    time_features_df['day'] = df.index.dayofweek
    time_features_df['day_str'] = df.index.strftime('%a')
    time_features_df['year_month'] = df.index.strftime('%Y_%m')

    # Reset the index so 'time' becomes a column, if needed
    time_features_df.reset_index(inplace=True)

    return time_features_df