"""
Base Model Interface
Abstract base class for all forecasting models
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional

import pandas as pd
import numpy as np


class BaseModel(ABC):
    """
    Abstract base class for all forecasting models.
    Defines common interface for training, prediction, and evaluation.
    """
    
    def __init__(self, name: str):
        """
        Initialize the model.
        
        Args:
            name: Model identifier name
        """
        self.name = name
        self.is_fitted = False
        self.train_dates = None
        self.train_metrics = {}
    
    @abstractmethod
    def fit(self, df: pd.DataFrame, target_col: str = 'close') -> 'BaseModel':
        """
        Train the model on historical data.
        
        Args:
            df: DataFrame with historical data
            target_col: Column to predict
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def predict(
        self,
        periods: int,
        return_confidence: bool = True
    ) -> pd.DataFrame:
        """
        Generate future predictions.
        
        Args:
            periods: Number of future periods to predict
            return_confidence: Whether to return confidence intervals
            
        Returns:
            DataFrame with predictions and optional confidence intervals
        """
        pass
    
    @abstractmethod
    def get_model_params(self) -> Dict[str, Any]:
        """
        Get model parameters and configuration.
        
        Returns:
            Dictionary of model parameters
        """
        pass
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            
        Returns:
            Dictionary of evaluation metrics
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        
        # Remove any NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        if len(y_true) == 0:
            return {'rmse': np.nan, 'mae': np.nan, 'mape': np.nan}
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        # MAPE (handle division by zero)
        non_zero_mask = y_true != 0
        if non_zero_mask.any():
            mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
        else:
            mape = np.nan
        
        return {
            'rmse': rmse,
            'mae': mae,
            'mape': mape
        }
    
    def prepare_train_test_split(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.8
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets (time-aware).
        
        Args:
            df: Full DataFrame
            train_ratio: Ratio of data to use for training
            
        Returns:
            Tuple of (train_df, test_df)
        """
        split_idx = int(len(df) * train_ratio)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        return train_df, test_df
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', is_fitted={self.is_fitted})"
