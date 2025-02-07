import unittest
import pandas as pd
from pathlib import Path
import os
from src.data.loader import DataLoader

class TestDataLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a test configuration
        cls.test_config = {
            'data': {
                'river_data_path': 'data/river_data/ea_gauge_data',
                'weather_data_path': 'data/weather_data',
                'merged_data_path': 'data/river_weather_merged_data'
            }
        }
        cls.data_loader = DataLoader(cls.test_config)

    def test_river_data_loading(self):
        # Get available stations
        stations = self.data_loader.get_available_stations()
        self.assertIsInstance(stations, list)
        self.assertTrue(len(stations) > 0, "No stations found in the river data directory")

        # Test loading data for the first available station
        first_station = stations[0]
        df = self.data_loader.load_river_data(first_station)
        
        # Check if the returned object is a DataFrame
        self.assertIsInstance(df, pd.DataFrame)
        
        # Check if the DataFrame has the expected index
        self.assertTrue(isinstance(df.index, pd.DatetimeIndex))
        
        # Check if DataFrame is not empty
        self.assertFalse(df.empty)
        
        # Check for required columns (based on actual CSV structure)
        expected_columns = {'measure', 'date', 'value', 'completeness', 'quality', 'qcode'}
        self.assertTrue(all(col in df.columns for col in expected_columns))

    def test_nonexistent_station(self):
        # Test loading data for a non-existent station
        with self.assertRaises(FileNotFoundError):
            self.data_loader.load_river_data("NonexistentStation")

    def test_get_available_stations(self):
        stations = self.data_loader.get_available_stations()
        
        # Check if stations is a non-empty list
        self.assertIsInstance(stations, list)
        self.assertTrue(len(stations) > 0)
        
        # Check if each station name is a string and properly formatted
        for station in stations:
            self.assertIsInstance(station, str)
            self.assertFalse(station.endswith('-level-15min-Qualified'))

if __name__ == '__main__':
    unittest.main()
