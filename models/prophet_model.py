"""
Prophet Model for Time Series Forecasting
Uses Facebook Prophet for trend and seasonality decomposition
"""

import logging
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np

from .base import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProphetModel(BaseModel):
    """
    Facebook Prophet model for stock price forecasting.
    Handles trends, seasonality, and provides built-in uncertainty intervals.
    """
    
    def __init__(
        self,
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = False
    ):
        """
        Initialize Prophet model.
        
        Args:
            changepoint_prior_scale: Flexibility of trend changes
            seasonality_prior_scale: Flexibility of seasonality
            yearly_seasonality: Include yearly patterns
            weekly_seasonality: Include weekly patterns
            daily_seasonality: Include daily patterns
        """
        super().__init__(name="Prophet")
        
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        
        self.model = None
        self.last_date = None
    
    def fit(self, df: pd.DataFrame, target_col: str = 'close') -> 'ProphetModel':
        """
        Train Prophet model on historical data.
        
        Args:
            df: DataFrame with 'date' and target column
            target_col: Column to predict (default: 'close')
            
        Returns:
            Self for method chaining
        """
        try:
            from prophet import Prophet
        except ImportError:
            raise ImportError("Prophet not installed. Run: pip install prophet")
        
        logger.info(f"Training Prophet model on {len(df)} data points")
        
        # Prepare data in Prophet format
        prophet_df = pd.DataFrame({
            'ds': pd.to_datetime(df['date']),
            'y': df[target_col].values
        })
        
        # Remove timezone if present
        if prophet_df['ds'].dt.tz is not None:
            prophet_df['ds'] = prophet_df['ds'].dt.tz_localize(None)
        
        # Initialize model
        self.model = Prophet(
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            uncertainty_samples=1000
        )
        
        # Fit model
        self.model.fit(prophet_df)
        
        self.last_date = prophet_df['ds'].max()
        self.is_fitted = True
        self.train_dates = (prophet_df['ds'].min(), prophet_df['ds'].max())
        
        logger.info("Prophet model training complete")
        
        return self
    
    def predict(
        self,
        periods: int = 30,
        return_confidence: bool = True
    ) -> pd.DataFrame:
        """
        Generate future predictions.
        
        Args:
            periods: Number of future days to predict
            return_confidence: Include confidence intervals
            
        Returns:
            DataFrame with predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        logger.info(f"Generating {periods}-day forecast with Prophet")
        
        # Create future dataframe
        future = self.model.make_future_dataframe(periods=periods)
        
        # Generate predictions
        forecast = self.model.predict(future)
        
        # Filter to future dates only
        future_mask = forecast['ds'] > self.last_date
        future_forecast = forecast[future_mask].copy()
        
        # Prepare output
        result = pd.DataFrame({
            'date': future_forecast['ds'],
            'predicted': future_forecast['yhat'],
            'lower_bound': future_forecast['yhat_lower'],
            'upper_bound': future_forecast['yhat_upper']
        })
        
        if not return_confidence:
            result = result[['date', 'predicted']]
        
        return result.reset_index(drop=True)
    
    def get_components(self) -> pd.DataFrame:
        """
        Get trend and seasonality components.
        
        Returns:
            DataFrame with component breakdown
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        future = self.model.make_future_dataframe(periods=0)
        forecast = self.model.predict(future)
        
        return forecast[['ds', 'trend', 'yearly', 'weekly']].copy()
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            'model_type': 'Prophet',
            'changepoint_prior_scale': self.changepoint_prior_scale,
            'seasonality_prior_scale': self.seasonality_prior_scale,
            'yearly_seasonality': self.yearly_seasonality,
            'weekly_seasonality': self.weekly_seasonality,
            'daily_seasonality': self.daily_seasonality
        }
