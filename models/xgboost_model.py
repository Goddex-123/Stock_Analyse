"""
XGBoost Model for Stock Price Prediction
Uses gradient boosting with technical indicators as features
"""

import logging
from typing import Dict, Any, List, Optional

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from .base import BaseModel
from config import XGBOOST_PARAMS, TRAIN_TEST_SPLIT

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class XGBoostModel(BaseModel):
    """
    XGBoost regressor for stock price prediction.
    Uses technical indicators as features for next-day prediction.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        feature_cols: Optional[List[str]] = None
    ):
        """
        Initialize XGBoost model.
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            feature_cols: List of feature columns to use
        """
        super().__init__(name="XGBoost")
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.feature_cols = feature_cols
        
        self.model = None
        self.scaler = StandardScaler()
        self.last_features = None
        self.last_close = None
        self.feature_importance = None
    
    def _prepare_features(
        self,
        df: pd.DataFrame,
        target_col: str = 'close'
    ) -> tuple:
        """
        Prepare features for training/prediction.
        
        Returns:
            Tuple of (X, y, feature_names)
        """
        df = df.copy()
        
        # Create target: next day's close price
        df['target'] = df[target_col].shift(-1)
        
        # Select feature columns
        if self.feature_cols is None:
            # Auto-select numeric columns except target and identifiers
            exclude_cols = ['date', 'target', target_col, 'open', 'high', 'low', 'volume',
                           'dividends', 'stock_splits']
            self.feature_cols = [col for col in df.columns 
                                if col not in exclude_cols 
                                and df[col].dtype in ['float64', 'int64']]
        
        # Filter to available columns
        available_cols = [col for col in self.feature_cols if col in df.columns]
        
        if len(available_cols) == 0:
            # Fallback: use price-based features
            available_cols = ['daily_return', 'log_return', 'volatility_20', 'rsi', 'macd']
            available_cols = [col for col in available_cols if col in df.columns]
        
        # Remove rows with NaN
        df_clean = df.dropna(subset=available_cols + ['target'])
        
        X = df_clean[available_cols].values
        y = df_clean['target'].values
        
        return X, y, available_cols
    
    def fit(self, df: pd.DataFrame, target_col: str = 'close') -> 'XGBoostModel':
        """
        Train XGBoost model on historical data.
        
        Args:
            df: DataFrame with features and target
            target_col: Column to predict
            
        Returns:
            Self for method chaining
        """
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")
        
        logger.info(f"Training XGBoost model on {len(df)} data points")
        
        # Prepare features
        X, y, feature_names = self._prepare_features(df, target_col)
        
        logger.info(f"Using {len(feature_names)} features: {feature_names[:5]}...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize and train model
        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_scaled, y)
        
        # Store feature importance
        self.feature_importance = dict(zip(feature_names, self.model.feature_importances_))
        
        # Store last row for future predictions
        df_clean = df.dropna(subset=feature_names)
        self.last_features = df_clean[feature_names].iloc[-1:].values
        self.last_close = df[target_col].iloc[-1]
        self.last_date = df['date'].iloc[-1]
        
        self.is_fitted = True
        self.train_dates = (df['date'].min(), df['date'].max())
        
        logger.info("XGBoost model training complete")
        
        return self
    
    def predict(
        self,
        periods: int = 30,
        return_confidence: bool = True
    ) -> pd.DataFrame:
        """
        Generate future predictions using recursive forecasting.
        
        Args:
            periods: Number of future days to predict
            return_confidence: Include confidence intervals
            
        Returns:
            DataFrame with predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        logger.info(f"Generating {periods}-day forecast with XGBoost")
        
        predictions = []
        dates = []
        
        # Simple recursive prediction
        # Note: This is an approximation - XGBoost predicts based on current features
        current_pred = self.last_close
        
        for i in range(periods):
            # Scale current features
            X_scaled = self.scaler.transform(self.last_features)
            
            # Predict next value
            next_pred = self.model.predict(X_scaled)[0]
            
            predictions.append(next_pred)
            
            # Next business day (skip weekends)
            next_date = self.last_date + pd.Timedelta(days=i+1)
            while next_date.weekday() >= 5:  # Skip weekends
                next_date += pd.Timedelta(days=1)
            dates.append(next_date)
            
            current_pred = next_pred
        
        predictions = np.array(predictions)
        
        # Calculate confidence intervals based on model uncertainty
        # Using historical prediction error as proxy
        result = pd.DataFrame({
            'date': dates,
            'predicted': predictions
        })
        
        if return_confidence:
            # Estimate uncertainty (increases with forecast horizon)
            base_std = np.std(predictions) * 0.1 if len(predictions) > 1 else self.last_close * 0.02
            uncertainties = [base_std * np.sqrt(i+1) for i in range(periods)]
            
            result['lower_bound'] = predictions - 1.96 * np.array(uncertainties)
            result['upper_bound'] = predictions + 1.96 * np.array(uncertainties)
        
        return result.reset_index(drop=True)
    
    def get_feature_importance(self, top_n: int = 10) -> Dict[str, float]:
        """
        Get top feature importances.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            Dictionary of feature -> importance
        """
        if self.feature_importance is None:
            return {}
        
        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return dict(sorted_features[:top_n])
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            'model_type': 'XGBoost',
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'n_features': len(self.feature_cols) if self.feature_cols else 0
        }
