"""
Model Evaluator Module
Compares and selects the best forecasting model
"""

import logging
from typing import Dict, Any, List, Tuple

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

from .base import BaseModel
from config import TRAIN_TEST_SPLIT

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluates and compares multiple forecasting models.
    Uses time-aware train-test split for proper evaluation.
    """
    
    def __init__(self, models: List[BaseModel]):
        """
        Initialize evaluator with models to compare.
        
        Args:
            models: List of model instances to evaluate
        """
        self.models = models
        self.results = {}
        self.best_model = None
        self.best_model_name = None
    
    def evaluate_models(
        self,
        df: pd.DataFrame,
        target_col: str = 'close',
        test_ratio: float = 0.2
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all models using time-aware split.
        
        Args:
            df: Full DataFrame with features
            target_col: Target column
            test_ratio: Ratio of data for testing
            
        Returns:
            Dictionary of model_name -> metrics
        """
        # Time-aware train-test split
        split_idx = int(len(df) * (1 - test_ratio))
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        logger.info(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
        
        y_test = test_df[target_col].values
        
        for model in self.models:
            try:
                logger.info(f"Evaluating {model.name}...")
                
                # Train model
                model.fit(train_df, target_col)
                
                # Predict for test period
                predictions = model.predict(periods=len(test_df), return_confidence=False)
                y_pred = predictions['predicted'].values
                
                # Calculate metrics
                metrics = self._calculate_metrics(y_test, y_pred)
                self.results[model.name] = metrics
                
                logger.info(f"{model.name} - RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}")
                
            except Exception as e:
                logger.error(f"Error evaluating {model.name}: {e}")
                self.results[model.name] = {
                    'rmse': np.inf,
                    'mae': np.inf,
                    'mape': np.inf,
                    'error': str(e)
                }
        
        # Select best model
        self._select_best_model()
        
        return self.results
    
    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        # Handle length mismatch
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        
        # Remove NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        if len(y_true) == 0:
            return {'rmse': np.inf, 'mae': np.inf, 'mape': np.inf}
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        # MAPE
        non_zero = y_true != 0
        if non_zero.any():
            mape = np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100
        else:
            mape = np.inf
        
        # Directional accuracy
        if len(y_true) > 1:
            actual_direction = np.sign(np.diff(y_true))
            pred_direction = np.sign(np.diff(y_pred))
            directional_accuracy = np.mean(actual_direction == pred_direction) * 100
        else:
            directional_accuracy = 50.0
        
        return {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'directional_accuracy': directional_accuracy
        }
    
    def _select_best_model(self):
        """Select the best performing model based on RMSE."""
        valid_results = {k: v for k, v in self.results.items() 
                        if 'error' not in v and v['rmse'] != np.inf}
        
        if not valid_results:
            logger.warning("No valid model results. Using first model.")
            self.best_model_name = self.models[0].name
            self.best_model = self.models[0]
            return
        
        # Select based on lowest RMSE
        self.best_model_name = min(valid_results, key=lambda k: valid_results[k]['rmse'])
        self.best_model = next(m for m in self.models if m.name == self.best_model_name)
        
        logger.info(f"Best model: {self.best_model_name}")
    
    def get_comparison_table(self) -> pd.DataFrame:
        """
        Get comparison table of all model results.
        
        Returns:
            DataFrame with model comparison
        """
        if not self.results:
            return pd.DataFrame()
        
        data = []
        for model_name, metrics in self.results.items():
            row = {
                'Model': model_name,
                'RMSE': metrics.get('rmse', np.nan),
                'MAE': metrics.get('mae', np.nan),
                'MAPE (%)': metrics.get('mape', np.nan),
                'Directional Accuracy (%)': metrics.get('directional_accuracy', np.nan)
            }
            if 'error' in metrics:
                row['Status'] = 'Error'
            else:
                row['Status'] = 'OK'
            data.append(row)
        
        return pd.DataFrame(data)
    
    def get_best_model(self) -> Tuple[BaseModel, str]:
        """
        Get the best performing model.
        
        Returns:
            Tuple of (model instance, model name)
        """
        if self.best_model is None:
            raise ValueError("Models must be evaluated first")
        
        return self.best_model, self.best_model_name
