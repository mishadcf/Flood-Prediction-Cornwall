from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional, List
import json
import yaml
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

@dataclass
class ExperimentResult:
    """Store experiment results and metadata"""
    experiment_id: str
    model_name: str
    model_params: Dict[str, Any]
    metrics: Dict[str, float]
    dataset_info: Dict[str, Any]
    timestamp: datetime
    description: str
    feature_importance: Optional[Dict[str, float]] = None
    artifacts_path: Optional[str] = None

class ExperimentTracker:
    """Track ML experiments and their results"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results_dir = Path(config.get('experiments', {}).get('results_dir', 'experiments'))
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.results_file = self.results_dir / 'experiment_results.csv'
        self._initialize_results_file()
    
    def _initialize_results_file(self):
        """Create results file if it doesn't exist"""
        if not self.results_file.exists():
            pd.DataFrame(columns=[
                'experiment_id', 'model_name', 'model_params', 'metrics',
                'dataset_info', 'timestamp', 'description', 
                'feature_importance', 'artifacts_path'
            ]).to_csv(self.results_file, index=False)
    
    def log_experiment(self, result: ExperimentResult):
        """Log experiment results"""
        # Convert result to dict
        result_dict = {
            'experiment_id': result.experiment_id,
            'model_name': result.model_name,
            'model_params': json.dumps(result.model_params),
            'metrics': json.dumps(result.metrics),
            'dataset_info': json.dumps(result.dataset_info),
            'timestamp': result.timestamp.isoformat(),
            'description': result.description,
            'feature_importance': json.dumps(result.feature_importance) if result.feature_importance else None,
            'artifacts_path': result.artifacts_path
        }
        
        # Append to CSV
        pd.DataFrame([result_dict]).to_csv(self.results_file, mode='a', header=False, index=False)
        
        # Save detailed results
        experiment_dir = self.results_dir / result.experiment_id
        experiment_dir.mkdir(exist_ok=True)
        
        with open(experiment_dir / 'metadata.yaml', 'w') as f:
            yaml.dump(result_dict, f)
    
    def get_best_experiments(self, metric: str, n: int = 5) -> pd.DataFrame:
        """Get the best N experiments based on a specific metric"""
        results = pd.read_csv(self.results_file)
        results['metrics'] = results['metrics'].apply(json.loads)
        results['metric_value'] = results['metrics'].apply(lambda x: x.get(metric, float('inf')))
        
        return results.nsmallest(n, 'metric_value')
    
    def get_experiment_details(self, experiment_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific experiment"""
        experiment_file = self.results_dir / experiment_id / 'metadata.yaml'
        if not experiment_file.exists():
            raise ValueError(f"Experiment {experiment_id} not found")
            
        with open(experiment_file, 'r') as f:
            return yaml.safe_load(f)

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate standard regression metrics"""
    return {
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'r2': float(r2_score(y_true, y_pred))
    }
