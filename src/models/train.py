import mlflow
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Union
import sys
import os
from abc import ABC, abstractmethod
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.data.loader import DataLoader
from src.models.metrics import evaluate_model
from src.utils.preprocessing import process_river, extract_time_features

class BaseModel(ABC):
    """Abstract base class for all models."""
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        """Fit the model to training data."""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions for X."""
        pass
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters for logging."""
        return self.__dict__

class ARIMAModel(BaseModel):
    """ARIMA model wrapper."""
    
    def __init__(self, order: tuple = (1,1,1), seasonal_order: tuple = (1,1,1,24)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        """Fit ARIMA model to time series data."""
        series = y if y is not None else X['value']
        self.model = ARIMA(series, 
                          order=self.order,
                          seasonal_order=self.seasonal_order).fit()
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions using fitted ARIMA model."""
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        return self.model.forecast(len(X))
    
    def get_params(self) -> Dict[str, Any]:
        """Get ARIMA parameters for logging."""
        return {
            'order': self.order,
            'seasonal_order': self.seasonal_order
        }

class ModelTrainer:
    """Handles model training and MLflow tracking."""
    
    def __init__(self, experiment_name: str = "flood-prediction"):
        self.experiment_name = experiment_name
        
        # Set MLflow tracking URI - using local directory by default
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        
        # Create or get experiment
        try:
            self.experiment_id = mlflow.create_experiment(
                self.experiment_name,
                artifact_location=str(project_root / "mlruns" / self.experiment_name)
            )
        except:
            self.experiment_id = mlflow.get_experiment_by_name(self.experiment_name).experiment_id
    
    def train_model(self, 
                   station_id: str,
                   model: BaseModel,
                   data: pd.DataFrame,
                   test_size: int = 168,  # 1 week of hourly data
                   feature_columns: Optional[list] = None) -> Dict[str, float]:
        """
        Train a model for a specific station with MLflow tracking.
        
        Args:
            station_id: River gauge station ID
            model: Instance of BaseModel
            data: DataFrame with datetime index
            test_size: Number of time steps for testing
            feature_columns: List of feature columns (for ML models)
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Split data into train/test
        train = data[:-test_size]
        test = data[-test_size:]
        
        # Start MLflow run
        with mlflow.start_run(experiment_id=self.experiment_id, 
                            run_name=f"{model.__class__.__name__}_{station_id}"):
            try:
                # Log parameters
                mlflow.log_params(model.get_params())
                mlflow.log_param("station_id", station_id)
                mlflow.log_param("model_type", model.__class__.__name__)
                
                # Prepare features if needed
                if feature_columns:
                    X_train = train[feature_columns]
                    y_train = train['value']
                    X_test = test[feature_columns]
                    y_test = test['value']
                    
                    # Fit and predict
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)
                else:
                    # For ARIMA, we only need the target variable
                    model.fit(train, train['value'])
                    predictions = model.predict(test)
                
                # Evaluate
                metrics = evaluate_model(test['value'].values, predictions)
                
                # Log metrics
                mlflow.log_metrics(metrics)
                
                # Log forecast plot
                self._log_forecast_plot(test['value'], predictions, station_id)
                
                return metrics
                
            except Exception as e:
                mlflow.log_param("error", str(e))
                raise e
    
    def _log_forecast_plot(self, actual: pd.Series, predicted: np.ndarray, station_id: str) -> None:
        """Create and log forecast comparison plot."""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 6))
        plt.plot(actual.index, actual.values, label='Actual', alpha=0.7)
        plt.plot(actual.index, predicted, label='Predicted', alpha=0.7)
        plt.title(f'Forecast vs Actual - Station {station_id}')
        plt.legend()
        plt.tight_layout()
        
        # Save plot
        plot_path = f'forecast_{station_id}.png'
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        plt.close()
        os.remove(plot_path)

def main():
    """Main training pipeline."""
    trainer = ModelTrainer()
    
    # Example ARIMA model parameters
    arima_params = {
        "order": (1, 1, 1),
        "seasonal_order": (1, 1, 1, 24)  # 24 for hourly seasonality
    }
    
    # Train ARIMA models for each station
    station_ids = ["station1", "station2"]  # Replace with your actual station IDs
    for station_id in station_ids:
        # Load and preprocess data
        data = DataLoader().load_station_data(station_id)
        data = process_river(data)
        
        # Train ARIMA baseline
        model = ARIMAModel(**arima_params)
        metrics = trainer.train_model(station_id, model, data)
        print(f"Station {station_id} metrics:", metrics)

if __name__ == "__main__":
    main()
