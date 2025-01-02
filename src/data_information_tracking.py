"""This script will track key information about the data for each river gauge at each transformation step.
    things that I need to keep track of in data dict:
RIVER GAUGES
----------------------------------------------------------------------------------------------
- gauge number
- geographical coordinates
- start date and end date of non-null data (measurements)
- basic summary statistics of raw data
- percentage and count of negative river gauge levels in raw data
- metadata of other anomalies (either due to weather extremes or mismeasurements)
- percentage of missing row measurements in raw data (counting nulls after using .asfreq('15min')
- basic summary statistics of data after applying step 1 transformation
- metadata of other anomalies (either due to weather extremes or mismeasurements) after step 1 
- step 2 cleaning will handle the missing 15min rows. So:
- basic summary statistics of data after applying step 2 transformation
- metadata of other anomalies (either due to weather extremes or mismeasurements) after step 2
    """
    
import pandas as pd
import numpy as np
import os
import json

river_data_station_info = 'data/river_data/station_info.csv'
river_data_raw_dir = 'data/river_data/highest_granularity/'
river_data_s1_dir = 'data/river_data/highest_granularity/cleaned/'
river_data_s1_plots_dir = 'data/river_data/highest_granularity/cleaned/plots/'





def build_station_data_dict(csv_path: str) -> pd.DataFrame:
    """
    Reads the CSV containing station info, parses the JSON in additionalDataObject
    (handling single quotes if present), and returns a DataFrame of 'key info'.
    """

    def safe_parse_json(s: str):
        if not isinstance(s, str):
            return {}
        # Convert single quotes to double quotes
        s_replaced = s.replace("'", '"')
        try:
            return json.loads(s_replaced)
        except json.JSONDecodeError:
            # Debug print if there's a row we can't parse
            print("JSON parse error. Original string:", s)
            return {}

    # Read the raw CSV
    df_raw = pd.read_csv(csv_path)
    
    # Check columns
    print("CSV columns:", df_raw.columns.tolist())
    
    # Make sure we actually have 'additionalDataObject' column
    if "additionalDataObject" not in df_raw.columns:
        raise ValueError("No 'additionalDataObject' column found in CSV!")
    
    # Apply safe_parse_json to the column
    df_raw["parsed_ado"] = df_raw["additionalDataObject"].apply(safe_parse_json)
    
    # Similarly handle gaugeList if it exists
    if "gaugeList" not in df_raw.columns:
        print("No 'gaugeList' column found; skipping gaugeList parsing...")
        df_raw["parsed_gauge_list"] = [{} for _ in range(len(df_raw))]
    else:
        def safe_parse_gauge_list(s: str):
            if not isinstance(s, str):
                return []
            s_replaced = s.replace("'", '"')
            try:
                return json.loads(s_replaced)
            except json.JSONDecodeError:
                print("GaugeList parse error. Original string:", s)
                return []
        
        df_raw["parsed_gauge_list"] = df_raw["gaugeList"].apply(safe_parse_gauge_list)

    # We'll store parsed results
    parsed_rows = []

    for _, row in df_raw.iterrows():
        # Basic columns
        station_id = row.get("id")
        name = row.get("name")
        latitude = row.get("latitude")
        longitude = row.get("longitude")
        station_owner = row.get("stationOwner")
        state = row.get("state")
        updated_time = row.get("updatedTime")

        # AdditionalDataObject dictionary
        ado = row["parsed_ado"]
        
        station_reference = ado.get("stationReference")
        catchment_name = ado.get("catchmentName")
        river_name = ado.get("riverName")
        elevation = ado.get("elevation")
        station_type = ado.get("stationType")
        region = ado.get("region")
        area_name = ado.get("areaName")
        date_open = ado.get("dateOpen")
        stage_datum = ado.get("stageDatum")

        # gaugeList dictionary (or list of dicts)
        gauge_list = row["parsed_gauge_list"] if isinstance(row["parsed_gauge_list"], list) else []
        gauge_entry = gauge_list[0] if len(gauge_list) > 0 else {}
        gauge_ado = gauge_entry.get("additionalDataObject", {})

        por_max = gauge_ado.get("porMax")
        date_por_max = gauge_ado.get("datePORMax")
        por_min = gauge_ado.get("porMin")
        date_por_min = gauge_ado.get("datePORMin")
        percentile95 = gauge_ado.get("percentile95")
        percentile5 = gauge_ado.get("percentile5")
        recent_highest = gauge_ado.get("recentHighest")
        date_highest = gauge_ado.get("dateHighest")
        units = gauge_entry.get("units")
        
        parsed_row = {
            "station_id": station_id,
            "station_reference": station_reference,
            "name": name,
            "latitude": latitude,
            "longitude": longitude,
            "station_owner": station_owner,
            "state": state,
            "updated_time": updated_time,
            "catchment_name": catchment_name,
            "river_name": river_name,
            "elevation": elevation,
            "station_type": station_type,
            "region": region,
            "area_name": area_name,
            "date_open": date_open,
            "stage_datum": stage_datum,

            "por_max": por_max,
            "date_por_max": date_por_max,
            "por_min": por_min,
            "date_por_min": date_por_min,
            "percentile95": percentile95,
            "percentile5": percentile5,
            "recent_highest": recent_highest,
            "date_highest": date_highest,
            "units": units
        }
        
        parsed_rows.append(parsed_row)

    df_parsed = pd.DataFrame(parsed_rows)
    return df_parsed
