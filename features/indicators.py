"""
Technical Indicators Module
Generates advanced features for stock analysis and prediction
"""

import numpy as np
import pandas as pd
from typing import List

from config import (
    RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    BOLLINGER_PERIOD, BOLLINGER_STD, ROLLING_WINDOWS
)


class TechnicalIndicators:
    """
    Generates technical indicators and features for stock data.
    """
    
    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all technical indicators to the DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all technical indicators added
        """
        df = df.copy()
        
        # Returns
        df = TechnicalIndicators.add_returns(df)
        
        # Rolling statistics
        df = TechnicalIndicators.add_rolling_statistics(df)
        
        # RSI
        df = TechnicalIndicators.add_rsi(df)
        
        # MACD
        df = TechnicalIndicators.add_macd(df)
        
        # Bollinger Bands
        df = TechnicalIndicators.add_bollinger_bands(df)
        
        # Moving Averages
        df = TechnicalIndicators.add_moving_averages(df)
        
        # Lag features
        df = TechnicalIndicators.add_lag_features(df)
        
        # Trend indicators
        df = TechnicalIndicators.add_trend_indicators(df)
        
        # Volume indicators
        df = TechnicalIndicators.add_volume_indicators(df)
        
        return df
    
    @staticmethod
    def add_returns(df: pd.DataFrame) -> pd.DataFrame:
        """Add daily returns and log returns."""
        df = df.copy()
        
        if 'close' in df.columns:
            # Simple returns
            df['daily_return'] = df['close'].pct_change()
            
            # Log returns
            df['log_return'] = np.log(df['close'] / df['close'].shift(1))
            
            # Multi-day returns
            df['return_5d'] = df['close'].pct_change(5)
            df['return_10d'] = df['close'].pct_change(10)
            df['return_20d'] = df['close'].pct_change(20)
        
        return df
    
    @staticmethod
    def add_rolling_statistics(
        df: pd.DataFrame,
        windows: List[int] = None
    ) -> pd.DataFrame:
        """Add rolling mean, std, and volatility."""
        df = df.copy()
        windows = windows or ROLLING_WINDOWS
        
        if 'close' in df.columns:
            for window in windows:
                # Rolling mean
                df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
                
                # Rolling std
                df[f'std_{window}'] = df['close'].rolling(window=window).std()
                
                # Rolling volatility (annualized)
                df[f'volatility_{window}'] = df['daily_return'].rolling(
                    window=window
                ).std() * np.sqrt(252)
                
                # Rolling min/max
                df[f'min_{window}'] = df['close'].rolling(window=window).min()
                df[f'max_{window}'] = df['close'].rolling(window=window).max()
        
        return df
    
    @staticmethod
    def add_rsi(df: pd.DataFrame, period: int = RSI_PERIOD) -> pd.DataFrame:
        """
        Add Relative Strength Index (RSI).
        RSI measures momentum on a scale of 0-100.
        """
        df = df.copy()
        
        if 'close' in df.columns:
            delta = df['close'].diff()
            
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            
            rs = avg_gain / avg_loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # RSI signals
            df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
            df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        
        return df
    
    @staticmethod
    def add_macd(
        df: pd.DataFrame,
        fast: int = MACD_FAST,
        slow: int = MACD_SLOW,
        signal: int = MACD_SIGNAL
    ) -> pd.DataFrame:
        """
        Add Moving Average Convergence Divergence (MACD).
        """
        df = df.copy()
        
        if 'close' in df.columns:
            # Calculate EMAs
            ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
            ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
            
            # MACD line
            df['macd'] = ema_fast - ema_slow
            
            # Signal line
            df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
            
            # MACD histogram
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # MACD crossover signals
            df['macd_bullish'] = (
                (df['macd'] > df['macd_signal']) & 
                (df['macd'].shift(1) <= df['macd_signal'].shift(1))
            ).astype(int)
            df['macd_bearish'] = (
                (df['macd'] < df['macd_signal']) & 
                (df['macd'].shift(1) >= df['macd_signal'].shift(1))
            ).astype(int)
        
        return df
    
    @staticmethod
    def add_bollinger_bands(
        df: pd.DataFrame,
        period: int = BOLLINGER_PERIOD,
        std_dev: float = BOLLINGER_STD
    ) -> pd.DataFrame:
        """
        Add Bollinger Bands.
        """
        df = df.copy()
        
        if 'close' in df.columns:
            # Middle band (SMA)
            df['bb_middle'] = df['close'].rolling(window=period).mean()
            
            # Standard deviation
            rolling_std = df['close'].rolling(window=period).std()
            
            # Upper and lower bands
            df['bb_upper'] = df['bb_middle'] + (std_dev * rolling_std)
            df['bb_lower'] = df['bb_middle'] - (std_dev * rolling_std)
            
            # Bollinger Band width
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            
            # %B indicator (position within bands)
            df['bb_percent'] = (
                (df['close'] - df['bb_lower']) / 
                (df['bb_upper'] - df['bb_lower'])
            )
        
        return df
    
    @staticmethod
    def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
        """Add exponential moving averages and crossover signals."""
        df = df.copy()
        
        if 'close' in df.columns:
            # EMAs
            df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
            df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
            df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
            
            # Golden cross / Death cross signals
            df['golden_cross'] = (
                (df['sma_50'] > df['sma_200']) & 
                (df['sma_50'].shift(1) <= df['sma_200'].shift(1))
            ).astype(int) if 'sma_200' in df.columns and 'sma_50' in df.columns else 0
            
            df['death_cross'] = (
                (df['sma_50'] < df['sma_200']) & 
                (df['sma_50'].shift(1) >= df['sma_200'].shift(1))
            ).astype(int) if 'sma_200' in df.columns and 'sma_50' in df.columns else 0
        
        return df
    
    @staticmethod
    def add_lag_features(
        df: pd.DataFrame,
        lags: List[int] = None
    ) -> pd.DataFrame:
        """Add lagged price and return features."""
        df = df.copy()
        lags = lags or [1, 2, 3, 5, 10, 20]
        
        if 'close' in df.columns:
            for lag in lags:
                df[f'close_lag_{lag}'] = df['close'].shift(lag)
                df[f'return_lag_{lag}'] = df['daily_return'].shift(lag)
        
        return df
    
    @staticmethod
    def add_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add trend strength and direction indicators."""
        df = df.copy()
        
        if 'close' in df.columns:
            # Price momentum
            df['momentum_10'] = df['close'] - df['close'].shift(10)
            df['momentum_20'] = df['close'] - df['close'].shift(20)
            
            # Rate of change
            df['roc_10'] = ((df['close'] - df['close'].shift(10)) / 
                           df['close'].shift(10)) * 100
            
            # Price relative to moving averages
            if 'sma_20' in df.columns:
                df['price_sma20_ratio'] = df['close'] / df['sma_20']
            if 'sma_50' in df.columns:
                df['price_sma50_ratio'] = df['close'] / df['sma_50']
            
            # Trend direction
            df['trend_short'] = np.where(
                df['close'] > df['close'].shift(5), 1,
                np.where(df['close'] < df['close'].shift(5), -1, 0)
            )
            df['trend_medium'] = np.where(
                df['close'] > df['close'].shift(20), 1,
                np.where(df['close'] < df['close'].shift(20), -1, 0)
            )
        
        return df
    
    @staticmethod
    def add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators."""
        df = df.copy()
        
        if 'volume' in df.columns:
            # Volume moving average
            df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
            
            # Volume ratio
            df['volume_ratio'] = df['volume'] / df['volume_sma_20']
            
            # On-Balance Volume (OBV)
            df['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()
            
            # Volume Rate of Change
            df['volume_roc'] = df['volume'].pct_change(10)
        
        return df
    
    @staticmethod
    def get_feature_columns(df: pd.DataFrame) -> List[str]:
        """Get list of feature columns suitable for ML models."""
        exclude_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 
                       'dividends', 'stock_splits']
        
        feature_cols = [col for col in df.columns 
                       if col not in exclude_cols and df[col].dtype in ['float64', 'int64']]
        
        return feature_cols
