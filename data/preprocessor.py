"""
Data Preprocessor Module
Handles data cleaning, missing value handling, and anomaly detection
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Preprocesses stock data for analysis and modeling.
    """
    
    @staticmethod
    def preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        """
        Main preprocessing pipeline for stock data.
        
        Args:
            df: Raw stock DataFrame
            
        Returns:
            Tuple of (processed DataFrame, preprocessing report)
        """
        report = {
            'original_rows': len(df),
            'missing_values_fixed': 0,
            'outliers_detected': 0,
            'duplicate_dates_removed': 0
        }
        
        df = df.copy()
        
        # Step 1: Ensure proper date formatting and sorting
        df = DataPreprocessor._handle_dates(df)
        
        # Step 2: Remove duplicate dates
        before_len = len(df)
        df = df.drop_duplicates(subset=['date'], keep='last')
        report['duplicate_dates_removed'] = before_len - len(df)
        
        # Step 3: Handle missing values
        df, missing_fixed = DataPreprocessor._handle_missing_values(df)
        report['missing_values_fixed'] = missing_fixed
        
        # Step 4: Detect and handle outliers
        df, outliers_detected = DataPreprocessor._handle_outliers(df)
        report['outliers_detected'] = outliers_detected
        
        # Step 5: Validate data integrity
        df = DataPreprocessor._validate_data(df)
        
        report['final_rows'] = len(df)
        
        logger.info(f"Preprocessing complete. Report: {report}")
        
        return df, report
    
    @staticmethod
    def _handle_dates(df: pd.DataFrame) -> pd.DataFrame:
        """Handle date column formatting and sorting."""
        # Ensure date column exists and is datetime
        if 'date' not in df.columns:
            if 'Date' in df.columns:
                df = df.rename(columns={'Date': 'date'})
            elif df.index.name and 'date' in df.index.name.lower():
                df = df.reset_index()
                df.columns = [c.lower() for c in df.columns]
        
        # Convert to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Remove timezone if present
        if df['date'].dt.tz is not None:
            df['date'] = df['date'].dt.tz_localize(None)
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        return df
    
    @staticmethod
    def _handle_missing_values(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """Handle missing values in the dataset."""
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_count = 0
        
        for col in numeric_cols:
            if col in df.columns:
                initial_missing = df[col].isna().sum()
                if initial_missing > 0:
                    # Forward fill then backward fill
                    df[col] = df[col].ffill().bfill()
                    missing_count += initial_missing
        
        return df, missing_count
    
    @staticmethod
    def _handle_outliers(
        df: pd.DataFrame,
        z_threshold: float = 4.0
    ) -> Tuple[pd.DataFrame, int]:
        """
        Detect and handle outliers using Z-score method.
        We use a high threshold (4.0) to only flag extreme outliers.
        """
        outlier_count = 0
        price_cols = ['open', 'high', 'low', 'close']
        
        for col in price_cols:
            if col in df.columns:
                # Calculate Z-scores
                mean = df[col].mean()
                std = df[col].std()
                
                if std > 0:
                    z_scores = np.abs((df[col] - mean) / std)
                    outliers = z_scores > z_threshold
                    outlier_count += outliers.sum()
                    
                    # Replace extreme outliers with interpolated values
                    if outliers.any():
                        df.loc[outliers, col] = np.nan
                        df[col] = df[col].interpolate(method='linear')
        
        return df, outlier_count
    
    @staticmethod
    def _validate_data(df: pd.DataFrame) -> pd.DataFrame:
        """Validate data integrity and fix common issues."""
        # Ensure high >= low
        if 'high' in df.columns and 'low' in df.columns:
            invalid_rows = df['high'] < df['low']
            if invalid_rows.any():
                # Swap high and low where invalid
                df.loc[invalid_rows, ['high', 'low']] = df.loc[invalid_rows, ['low', 'high']].values
        
        # Ensure close is within high-low range
        if all(col in df.columns for col in ['close', 'high', 'low']):
            df['close'] = df['close'].clip(lower=df['low'], upper=df['high'])
        
        # Ensure volume is non-negative
        if 'volume' in df.columns:
            df['volume'] = df['volume'].clip(lower=0)
        
        # Ensure prices are positive
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                df[col] = df[col].clip(lower=0.01)
        
        return df
    
    @staticmethod
    def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features from the date column."""
        df = df.copy()
        
        if 'date' in df.columns:
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['day'] = df['date'].dt.day
            df['day_of_week'] = df['date'].dt.dayofweek
            df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
            df['quarter'] = df['date'].dt.quarter
            
            # Cyclical encoding for periodic features
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
