import pandas as pd
import os
from typing import Dict, Any, Optional, List
from pathlib import Path

class DataLoader:
    """
    Handles all data loading operations for the flood prediction system.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data loader with configuration.
        
        Args:
            config: Dictionary containing data configuration
        """
        self.config = config
        self.river_data_path = Path(config['data']['river_data_path'])
        self.weather_data_path = Path(config['data']['weather_data_path'])
        self.merged_data_path = Path(config['data']['merged_data_path'])
        
    def load_river_data(self, station_name: str) -> pd.DataFrame:
        """
        Load river gauge data for a specific station.
        
        Args:
            station_name: Name of the river gauge station (e.g., 'Boscastle-New-Mills')
            
        Returns:
            DataFrame containing river gauge data
        
        Raises:
            FileNotFoundError: If the station data file is not found
        """
        # Convert station name to expected file format
        file_name = f"{station_name}-level-15min-Qualified.csv"
        file_path = self.river_data_path / file_name
        
        if not file_path.exists():
            raise FileNotFoundError(f"No data file found for station: {station_name}")
            
        # Read and process the data
        df = pd.read_csv(file_path)
        df['time'] = pd.to_datetime(df['dateTime'])
        df = df.drop('dateTime', axis=1)
        df.set_index('time', inplace=True)
        return df
        
    def load_weather_data(self) -> pd.DataFrame:
        """
        Load weather data.
        
        Returns:
            DataFrame containing weather data
        """
        weather_files = list(self.weather_data_path.glob("*.csv"))
        dfs = []
        
        for file in weather_files:
            df = pd.read_csv(file)
            df['time'] = pd.to_datetime(df['time'])
            dfs.append(df)
            
        return pd.concat(dfs, axis=0).drop_duplicates()
    
    def load_merged_data(self, station_name: str) -> pd.DataFrame:
        """
        Load merged river and weather data for a specific station.
        
        Args:
            station_name: Name of the river gauge station
            
        Returns:
            DataFrame containing merged data
        """
        # Convert station name to a format used in merged files
        station_name_formatted = station_name.replace('-', '_').lower()
        file_path = self.merged_data_path / f"merged_{station_name_formatted}.csv"
        
        if not file_path.exists():
            raise FileNotFoundError(f"No merged data file found for station: {station_name}")
            
        df = pd.read_csv(file_path)
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        return df
    
    def get_available_stations(self) -> List[str]:
        """
        Get list of available station names.
        
        Returns:
            List of station names (without the '-level-15min-Qualified.csv' suffix)
        """
        files = self.river_data_path.glob("*-level-15min-Qualified.csv")
        return [f.stem.replace('-level-15min-Qualified', '') for f in files]
    
    def station_name_to_id(self, station_name: str) -> Optional[str]:
        """
        Convert a station name to its corresponding ID if available.
        This method can be implemented when you have a mapping between names and IDs.
        
        Args:
            station_name: Name of the station
            
        Returns:
            Station ID if available, None otherwise
        """
        # TODO: Implement mapping between station names and IDs if needed
        pass
