import pandas as pd
import json
import requests
import os
from meteostat import Hourly, Daily, Monthly, Stations
from datetime import datetime
import yaml 
import openmeteo_requests
import requests_cache
from retry_requests import retry
import os
import time 


# placeholder for later
if __name__ == '__main__':
    pass


headers = {
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Origin": "https://www.gaugemap.co.uk",
    "Referer": "https://www.gaugemap.co.uk/",
    "SessionHeaderId": "03173723-4dea-4c81-8d8e-5c808698384b",  # May Need to rotate session/ user-agent headers
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
}

west_cornwall_url = "https://riverlevelsapi.azurewebsites.net/TimeSeries/GetTimeSeriesStationsByCatchmentName/?catchmentName=West+Cornwall"

north_cornwall_url = "https://riverlevelsapi.azurewebsites.net/TimeSeries/GetTimeSeriesStationsByCatchmentName/?catchmentName=North+Cornwall"

def load_credentials():
    # Load the credentials from secrets.yaml
    with open("secrets.yaml", "r") as file:
        secrets = yaml.safe_load(file)

    # Access API keys
    river_gauge_api_key = secrets["api_keys"]["river_gauge_api"]
    weather_api_key = secrets["api_keys"]["noaa_api"]

    return river_gauge_api_key, weather_api_key

#TODO: adapt secrets to enable various choices of API for weather data, potentially river data


def get_top_level_info_river_gauges(
    urls: list = [north_cornwall_url, west_cornwall_url], headers: dict = headers
) -> pd.DataFrame:
    """
    The function `get_top_level_info` retrieves data from specified URLs using provided headers and
    returns a concatenated DataFrame.

    :param urls: The `urls` parameter in the `get_top_level_info` function is a list of URLs that the
    function will make requests to in order to retrieve data. In this case, the default URLs provided
    are `north_cornwall_url` and `west_cornwall_url`
    :type urls: list
    :param headers: Headers typically contain information such as user-agent, content type, and
    authorization details that are sent along with a request to a web server. They help identify the
    type of content being sent and authenticate the request. In the context of the provided code
    snippet, the `headers` parameter is likely a dictionary containing
    :type headers: dict
    :return: A pandas DataFrame containing the top-level information from the URLs provided in the
    `urls` list, fetched using the specified headers.
    """

    results = pd.DataFrame()

    for url in urls:
        response = requests.get(url, headers=headers)
        data = json.loads(response.text)
        data = pd.json_normalize(data)
        results = pd.concat([results, data], ignore_index=True)

    return results


# next : pull river data from the IDs listed in the result (as above)

def fetch_and_save_river_data(station_ids, start_date, end_date, smoothing=2, destination="data/river_data"):
    """
    Fetch river gauge measurements and save them to individual CSV files.

    Args:
    station_ids (list): A list of station IDs to fetch data for.
    start_date (str): The start date in ISO 8601 format (e.g., '2013-04-30T23:00:00').
    end_date (str): The end date in ISO 8601 format (e.g., '2023-08-31T22:59:59').
    smoothing (int, optional): Smoothing parameter for the API. Default is 2.
    destination (str, optional): The destination directory where the CSV files will be saved. Default is 'data/river_data'.

    Returns:
    None
    """
    base_url = "https://riverlevelsapi.azurewebsites.net/TimeSeries/GetTimeSeriesDatapointsDateTime/?stationId={station_id}&dataType=3&endTime={end_time}&startTime={start_time}&smoothing={smoothing}"

    headers = {
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Origin": "https://www.gaugemap.co.uk",
        "Referer": "https://www.gaugemap.co.uk/",
        "SessionHeaderId": "03173723-4dea-4c81-8d8e-5c808698384b",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    }

    # Ensure the destination directory exists
    if not os.path.exists(destination):
        os.makedirs(destination)

    for station_id in station_ids:
        formatted_url = base_url.format(
            station_id=station_id,
            end_time=end_date,
            start_time=start_date,
            smoothing=smoothing,
        )
        response = requests.get(formatted_url, headers=headers)
        if response.status_code == 200:
            data = json.loads(response.text)
            df = pd.json_normalize(data)  # Create DataFrame from JSON
            # Save the file in the specified destination
            file_path = os.path.join(destination, f"station_{station_id}.csv")
            df.to_csv(file_path, index=False)
            print(f"Data for station {station_id} saved successfully in {file_path}.")
        else:
            print(f"Failed to fetch data for station {station_id}. Status code: {response.status_code}")



def get_weather_station_info_meteostat(coords=None):
    """
    Fetches information about the 10 nearest weather stations to the specified coordinates.
    If no coordinates are provided, defaults to the coordinates for Cornwall - this is what we're modelling.

    Parameters:
    coords (tuple, optional): A tuple containing the latitude (float) and longitude (float) of the desired location.

    Returns:
    DataFrame: A DataFrame containing information about each of the 10 closest weather stations.
    """
    # Default coordinates to Cornwall if none provided
    if coords is None:
        coords = (50.2660, -5.0527)  # Latitude and Longitude of Cornwall
    elif not isinstance(coords, tuple) or len(coords) != 2:
        raise ValueError(
            "Coordinates must be provided as a tuple of (latitude, longitude)."
        )

    stations = Stations()
    nearby_stations = stations.nearby(lat=coords[0], lon=coords[1])

    try:
        station_info = nearby_stations.fetch(10)
        return station_info
    except Exception as e:
        print(f"Failed to pull data: {e}")
        return None



def fetch_weather_data_meteostat(station_id: str, dates: tuple = None, granularity_class=Hourly):
    """
    Fetches weather data for a specified station ID within a given date range with specified granularity.

    Parameters:
        station_id (str): Unique identifier of the weather station.
        dates (tuple, optional): Tuple containing start and end datetime objects.
                                 If no dates are provided, it defaults from March 4, 2014, at 6:15 AM to today.
        granularity_class: Class used to fetch weather data (e.g., Hourly, Daily, Monthly).
                           Defaults to Hourly.

    Returns:
        DataFrame: Weather data for the specified station ID and dates, or None if an error occurs.
    """
    # Default date range set if no dates are provided
    if dates is None:
        dates = (datetime(2000, 3, 4, 6, 15, 0), datetime.today())

    try:
        # Initialize the granularity class with the specified parameters
        weather_data = granularity_class(station_id, start=dates[0], end=dates[1])
        fetched_data = weather_data.fetch()
        return fetched_data
    except Exception as e:
        print(f"Failed to fetch data: {e}")
        return None


def pull_noaa_weather_data(station_id, start_date, end_date, token, datatypes):
    url = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"
    params = {
        "datasetid": "GHCND",
        "stationid": f"GHCND:{station_id}",
        "startdate": start_date,
        "enddate": end_date,
        "limit": 1000,
        "datatypeid": datatypes,  # Pass a list of datatype IDs
        "units": "metric"
    }
    headers = {"token": token}

    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        return response.json()  # Returns JSON data
    else:
        print(f"Error: {response.status_code}")
        return None

def fetch_weather_data_over_years_noaa(station_id, start_year, end_year, token, datatypes):
    all_data = []
    for year in range(start_year, end_year + 1):
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        print(f"Fetching data for {year}...")
        weather_data = pull_noaa_weather_data(station_id, start_date, end_date, token, datatypes)
        
        if weather_data and 'results' in weather_data:
            all_data.extend(weather_data['results'])  # Combine the yearly data
    
    return all_data


def fetch_and_save_weather_data_2(station_coordinates, url = "https://archive-api.open-meteo.com/v1/archive", params = {}):
    """
    Fetch and save weather data from Jan 1, 2016, to the current date for each station.
    Saves each station's data as a CSV, skips already downloaded stations, and pauses to respect rate limits.
    """
    # Define the start and end dates within the function
    current_date = datetime.now()
    start_date_2016 = datetime(2016, 1, 1)
    
    
    # Setup the Open-Meteo API client with caching and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)
    
    
    # Create the output directory if it doesn't exist
    output_dir = "data/weather_data"
    os.makedirs(output_dir, exist_ok=True)



    # Set date parameters for the allowed range (Jan 1, 2016 to today)
    params["start_date"] = start_date_2016.strftime('%Y-%m-%d')
    params["end_date"] = current_date.strftime('%Y-%m-%d')
    params["hourly"] = ["temperature_2m", "relative_humidity_2m", "rain", "pressure_msl", "surface_pressure", "soil_moisture_0_to_7cm"]
    
    request_count = 0
    max_requests_per_minute = 5  # Set this based on your API limit

    for station_name, (lat, lon) in station_coordinates.items():
        # Construct the CSV file path
        csv_path = os.path.join(output_dir, f"{station_name}_nearest_weather_station_openmeteo.csv")

        # Skip if the file already exists
        if os.path.exists(csv_path):
            print(f"Data for {station_name} already exists. Skipping...")
            continue

        params["latitude"], params["longitude"] = lat, lon
        try:
            # API call to retrieve weather data
            responses = openmeteo.weather_api(url, params=params)
            if responses:
                response = responses[0]

                # Get weather station coordinates
                station_lat = response.Latitude()
                station_lon = response.Longitude()
                print(f"{station_name}: Closest weather station at ({station_lat}, {station_lon})")

                # Access hourly data and create a DataFrame
                hourly = response.Hourly()
                hourly_data = {
                    "date": pd.date_range(
                        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                        freq=pd.Timedelta(seconds=hourly.Interval()),
                        inclusive="left"
                    ),
                    "temperature_2m": hourly.Variables(0).ValuesAsNumpy(),
                    "relative_humidity_2m": hourly.Variables(1).ValuesAsNumpy(),
                    "rain": hourly.Variables(2).ValuesAsNumpy(),
                    "pressure_msl": hourly.Variables(3).ValuesAsNumpy(),
                    "surface_pressure": hourly.Variables(4).ValuesAsNumpy(),
                    "soil_moisture_0_to_7cm": hourly.Variables(5).ValuesAsNumpy(),
                    "weather_station_latitude": [station_lat] * len(hourly.Variables(0).ValuesAsNumpy()),
                    "weather_station_longitude": [station_lon] * len(hourly.Variables(0).ValuesAsNumpy())
                }

                # Save data to a CSV file
                df = pd.DataFrame(data=hourly_data)
                df.to_csv(csv_path, index=False)
                print(f"Data for {station_name} saved to {csv_path}")

                # Increment the request count
                request_count += 1

                # Pause to respect the rate limit
                if request_count >= max_requests_per_minute:
                    print("Rate limit reached, pausing for 1 minute...")
                    time.sleep(60)
                    request_count = 0  # Reset counter after pause

        except Exception as e:
            print(f"Error fetching data for {station_name} ({lat}, {lon}): {e}")
            print("Waiting for 1 minute before retrying...")
            time.sleep(60)  # Wait before retrying
        except Exception as e:
            print(f"Error fetching data for {station_name} ({lat}, {lon}): {e}")

# # Example usage
# url = "https://archive-api.open-meteo.com/v1/archive"
# params = {}

# # Dictionary with station names and coordinates
# station_coordinates = {
#     "station_1": (50.47965214144801, -4.795632626910022),
#     "station_2": (50.806426509528855, -4.53645942100034),
#     "station_3": (50.84448721946478, -4.509159997600046),
# }

# # Fetch and save data for each station
# fetch_and_save_weather_data(url, params, station_coordinates)

# # Example usage
# api_key = "12345"  # Replace with your NOAA API key
# station_id = "UK000003808"  # Cornwall station ID
# start_year = 2013
# end_year = 2023
# datatypes = ["PRCP", "TMAX", "TMIN", "AWND", "WSF2"]  # Add more data types if needed

# # Fetch data across multiple years
# all_weather_data = fetch_weather_data_over_years(station_id, start_year, end_year, api_key, datatypes)

# # Convert the results into a DataFrame for easier analysis
# if all_weather_data:
#     df = pd.DataFrame(all_weather_data)
#     print(df.head())

#TODO: split weather and river data functions into classes, allowing for options with different weather servies and perhaps river gauge services. 

