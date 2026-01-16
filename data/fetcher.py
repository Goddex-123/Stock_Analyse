"""
Stock Data Fetcher Module
Handles fetching historical stock data from Yahoo Finance
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple

import pandas as pd
import yfinance as yf
import streamlit as st

from config import DEFAULT_YEARS_HISTORY

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockDataFetcher:
    """
    Fetches stock data from Yahoo Finance with caching and error handling.
    """
    
    def __init__(self):
        self.cache = {}
    
    @staticmethod
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def fetch_stock_data(
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        years: int = DEFAULT_YEARS_HISTORY
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Fetch historical stock data from Yahoo Finance.
        
        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'RELIANCE.NS')
            start_date: Start date for data fetching
            end_date: End date for data fetching
            years: Number of years of historical data (used if dates not specified)
        
        Returns:
            Tuple of (DataFrame with stock data, error message if any)
        """
        try:
            # Set default date range
            if end_date is None:
                end_date = datetime.now()
            if start_date is None:
                start_date = end_date - timedelta(days=years * 365)
            
            logger.info(f"Fetching data for {symbol} from {start_date} to {end_date}")
            
            # Create ticker object
            ticker = yf.Ticker(symbol)
            
            # Fetch historical data
            df = ticker.history(
                start=start_date,
                end=end_date,
                auto_adjust=True
            )
            
            # Validate data
            if df.empty:
                return None, f"No data found for symbol '{symbol}'. Please check if the symbol is valid."
            
            if len(df) < 30:
                return None, f"Insufficient data for '{symbol}'. Only {len(df)} days available. Need at least 30 days."
            
            # Reset index to make Date a column
            df = df.reset_index()
            
            # Standardize column names
            df.columns = [col.lower().replace(' ', '_') for col in df.columns]
            
            # Ensure date column is datetime
            df['date'] = pd.to_datetime(df['date'])
            
            logger.info(f"Successfully fetched {len(df)} rows for {symbol}")
            
            return df, None
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error fetching data for {symbol}: {error_msg}")
            
            if "No data found" in error_msg:
                return None, f"Invalid symbol '{symbol}'. Please enter a valid stock ticker."
            elif "ConnectionError" in error_msg or "timeout" in error_msg.lower():
                return None, "Network error. Please check your internet connection and try again."
            else:
                return None, f"Error fetching data: {error_msg}"
    
    @staticmethod
    def get_stock_info(symbol: str) -> Optional[dict]:
        """
        Get basic information about a stock.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary with stock info or None if error
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'name': info.get('longName', symbol),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 'N/A'),
                'currency': info.get('currency', 'USD'),
                'exchange': info.get('exchange', 'N/A'),
                'current_price': info.get('currentPrice', info.get('regularMarketPrice', 'N/A')),
                'previous_close': info.get('previousClose', 'N/A'),
                '52_week_high': info.get('fiftyTwoWeekHigh', 'N/A'),
                '52_week_low': info.get('fiftyTwoWeekLow', 'N/A')
            }
        except Exception as e:
            logger.error(f"Error fetching stock info for {symbol}: {e}")
            return None
    
    @staticmethod
    def validate_symbol(symbol: str) -> Tuple[bool, str]:
        """
        Validate if a stock symbol exists.
        
        Args:
            symbol: Stock ticker symbol to validate
            
        Returns:
            Tuple of (is_valid, message)
        """
        if not symbol or not symbol.strip():
            return False, "Please enter a stock symbol."
        
        symbol = symbol.strip().upper()
        
        try:
            ticker = yf.Ticker(symbol)
            # Try to fetch minimal data to validate
            hist = ticker.history(period="5d")
            
            if hist.empty:
                return False, f"Symbol '{symbol}' not found. Please check the ticker symbol."
            
            return True, f"Symbol '{symbol}' is valid."
            
        except Exception as e:
            return False, f"Error validating symbol: {str(e)}"
