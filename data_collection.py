import pandas as pd
import json
import requests

headers = {
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Origin": "https://www.gaugemap.co.uk",
    "Referer": "https://www.gaugemap.co.uk/",
    "SessionHeaderId": "03173723-4dea-4c81-8d8e-5c808698384b",  # Need to rotate session/ user-agent headers
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
}

west_cornwall_url = "https://riverlevelsapi.azurewebsites.net/TimeSeries/GetTimeSeriesStationsByCatchmentName/?catchmentName=West+Cornwall"

north_cornwall_url = "https://riverlevelsapi.azurewebsites.net/TimeSeries/GetTimeSeriesStationsByCatchmentName/?catchmentName=North+Cornwall"


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


def fetch_and_save_river_data(
    station_ids, start_date, end_date, smoothing=2, headers=None
):
    """
    Fetch river gauge measurements and save them to individual CSV files.

    Args:
    station_ids (list): A list of station IDs to fetch data for.
    start_date (str): The start date in ISO 8601 format (e.g., '2013-04-30T23:00:00').
    end_date (str): The end date in ISO 8601 format (e.g., '2023-08-31T22:59:59').
    smoothing (int, optional): Smoothing parameter for the API. Default is 2.
    headers (dict, optional): HTTP headers to send with the requests.

    Returns:
    None
    """
    base_url = "https://riverlevelsapi.azurewebsites.net/TimeSeries/GetTimeSeriesDatapointsDateTime/?stationId={station_id}&dataType=3&endTime={end_time}&startTime={start_time}&smoothing={smoothing}"

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
            df = pd.json_normalize(data)  # create DataFrame from JSON
            df.to_csv(
                f"station_{station_id}.csv", index=False, path="get_river_data/data"
            )  # save DataFrame to a CSV file
            print(f"Data for station {station_id} saved successfully.")
        else:
            print(
                f"Request failed for station {station_id}, status code: {response.status_code}"
            )


if __name__ == "__main__":
    pass
