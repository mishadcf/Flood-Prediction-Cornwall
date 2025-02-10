import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate standard regression metrics for flood prediction models.
    
    Args:
        y_true: Ground truth values
        y_pred: Model predictions
        
    Returns:
        Dictionary containing RMSE, MAE, and R² metrics
    """
    return {
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'r2': float(r2_score(y_true, y_pred))
    }
