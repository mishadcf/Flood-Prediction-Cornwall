from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any, Union, Tuple
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

class BaseModel(ABC):
    """
    Abstract base class for all models in the flood prediction system.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the base model with configuration.
        
        Args:
            config: Dictionary containing model configuration
        """
        self.config = config
        self.model = None
        
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train the model.
        
        Args:
            X: Feature DataFrame
            y: Target series
        """
        pass
        
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of predictions
        """
        pass
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate the model using multiple metrics.
        
        Args:
            X: Feature DataFrame
            y: True target values
            
        Returns:
            Dictionary of metric names and values
        """
        predictions = self.predict(X)
        return {
            'rmse': np.sqrt(mean_squared_error(y, predictions)),
            'mae': mean_absolute_error(y, predictions),
            'r2': r2_score(y, predictions)
        }
    
    def save(self, path: str) -> None:
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model
        """
        raise NotImplementedError
        
    def load(self, path: str) -> None:
        """
        Load the model from disk.
        
        Args:
            path: Path to load the model from
        """
        raise NotImplementedError
