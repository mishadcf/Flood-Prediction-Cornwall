import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler
from .train import BaseModel

class TimeSeriesDataset(Dataset):
    """Dataset for sequence models (LSTM, Transformer)."""
    def __init__(self, 
                 data: pd.DataFrame, 
                 sequence_length: int,
                 target_column: str = 'value',
                 feature_columns: Optional[List[str]] = None):
        self.sequence_length = sequence_length
        self.feature_columns = feature_columns or [target_column]
        
        # Prepare sequences
        self.X, self.y = self._prepare_sequences(data, target_column)
        
    def _prepare_sequences(self, 
                          data: pd.DataFrame, 
                          target_column: str) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for training."""
        X, y = [], []
        
        features = data[self.feature_columns].values
        for i in range(len(data) - self.sequence_length):
            X.append(features[i:(i + self.sequence_length)])
            y.append(data[target_column].iloc[i + self.sequence_length])
            
        return np.array(X), np.array(y)
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (torch.FloatTensor(self.X[idx]), 
                torch.FloatTensor([self.y[idx]]))

class XGBoostModel(BaseModel):
    """XGBoost wrapper for time series forecasting."""
    
    def __init__(self, **params):
        self.model = xgb.XGBRegressor(**params)
        self.params = params
        self.feature_scaler = StandardScaler()
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        # Scale features
        X_scaled = self.feature_scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.feature_scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_params(self) -> Dict[str, Any]:
        return self.params

class CatBoostModel(BaseModel):
    """CatBoost wrapper for time series forecasting."""
    
    def __init__(self, **params):
        self.model = CatBoostRegressor(**params)
        self.params = params
        self.feature_scaler = StandardScaler()
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        X_scaled = self.feature_scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.feature_scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_params(self) -> Dict[str, Any]:
        return self.params

class LSTMModel(BaseModel):
    """LSTM model for time series forecasting."""
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 dropout: float = 0.1,
                 batch_size: int = 32,
                 epochs: int = 100,
                 learning_rate: float = 0.001,
                 sequence_length: int = 24):
        super().__init__()
        self.params = locals()
        del self.params['self']
        
        self.model = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.linear = nn.Linear(hidden_size, 1)
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        # Scale data
        X_scaled = self.scaler.fit_transform(X)
        
        # Create dataset
        dataset = TimeSeriesDataset(
            pd.DataFrame(X_scaled, columns=X.columns, index=X.index),
            sequence_length=self.sequence_length,
            feature_columns=X.columns.tolist()
        )
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        
        # Training
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate
        )
        criterion = nn.MSELoss()
        
        self.model.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                
                # Forward pass
                lstm_out, _ = self.model(batch_X)
                predictions = self.linear(lstm_out[:, -1, :])
                
                # Compute loss
                loss = criterion(predictions, batch_y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        
        dataset = TimeSeriesDataset(
            pd.DataFrame(X_scaled, columns=X.columns, index=X.index),
            sequence_length=self.sequence_length,
            feature_columns=X.columns.tolist()
        )
        dataloader = DataLoader(dataset, batch_size=len(dataset))
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = next(iter(dataloader))[0]
            lstm_out, _ = self.model(X_tensor)
            predictions = self.linear(lstm_out[:, -1, :])
            
        return predictions.numpy().flatten()
    
    def get_params(self) -> Dict[str, Any]:
        return self.params

class TransformerModel(BaseModel):
    """Transformer model for time series forecasting."""
    
    def __init__(self,
                 input_size: int,
                 d_model: int = 64,
                 nhead: int = 8,
                 num_encoder_layers: int = 3,
                 dim_feedforward: int = 256,
                 dropout: float = 0.1,
                 batch_size: int = 32,
                 epochs: int = 100,
                 learning_rate: float = 0.001,
                 sequence_length: int = 168):
        super().__init__()
        self.params = locals()
        del self.params['self']
        
        self.model = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.input_projection = nn.Linear(input_size, d_model)
        self.output_projection = nn.Linear(d_model, 1)
        
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        # Scale data
        X_scaled = self.scaler.fit_transform(X)
        
        # Create dataset
        dataset = TimeSeriesDataset(
            pd.DataFrame(X_scaled, columns=X.columns, index=X.index),
            sequence_length=self.sequence_length,
            feature_columns=X.columns.tolist()
        )
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        
        # Training
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate
        )
        criterion = nn.MSELoss()
        
        self.model.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                
                # Project input to d_model dimensions
                src = self.input_projection(batch_X)
                tgt = src.clone()
                
                # Create masks
                src_mask = self._generate_square_subsequent_mask(src.size(1))
                tgt_mask = self._generate_square_subsequent_mask(tgt.size(1))
                
                # Forward pass
                output = self.model(
                    src, tgt,
                    src_mask=src_mask,
                    tgt_mask=tgt_mask
                )
                predictions = self.output_projection(output[:, -1, :])
                
                # Compute loss
                loss = criterion(predictions, batch_y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        
        dataset = TimeSeriesDataset(
            pd.DataFrame(X_scaled, columns=X.columns, index=X.index),
            sequence_length=self.sequence_length,
            feature_columns=X.columns.tolist()
        )
        dataloader = DataLoader(dataset, batch_size=len(dataset))
        
        self.model.eval()
        with torch.no_grad():
            batch_X = next(iter(dataloader))[0]
            src = self.input_projection(batch_X)
            tgt = src.clone()
            
            src_mask = self._generate_square_subsequent_mask(src.size(1))
            tgt_mask = self._generate_square_subsequent_mask(tgt.size(1))
            
            output = self.model(
                src, tgt,
                src_mask=src_mask,
                tgt_mask=tgt_mask
            )
            predictions = self.output_projection(output[:, -1, :])
            
        return predictions.numpy().flatten()
    
    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate mask for transformer self-attention."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf'))
        mask = mask.masked_fill(mask == 1, float(0.0))
        return mask
    
    def get_params(self) -> Dict[str, Any]:
        return self.params
