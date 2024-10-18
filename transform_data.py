import pandas as pd
from ast import literal_eval
import json
import os


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
