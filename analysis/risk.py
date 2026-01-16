"""
Risk Analysis Module
Provides risk metrics and uncertainty analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from scipy import stats

from config import VAR_CONFIDENCE_LEVEL, VOLATILITY_WINDOW


class RiskAnalyzer:
    """
    Analyzes risk metrics and uncertainty for stock predictions.
    """
    
    @staticmethod
    def calculate_volatility_metrics(df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate various volatility metrics.
        
        Args:
            df: DataFrame with price data and returns
            
        Returns:
            Dictionary of volatility metrics
        """
        returns = df['daily_return'].dropna()
        
        metrics = {
            # Historical Volatility
            'daily_volatility': returns.std(),
            'annualized_volatility': returns.std() * np.sqrt(252),
            
            # Recent volatility (last 20 days)
            'recent_volatility': returns.tail(VOLATILITY_WINDOW).std() * np.sqrt(252),
            
            # Volatility percentile (how current vol compares to historical)
            'volatility_percentile': RiskAnalyzer._calculate_volatility_percentile(returns),
            
            # Average True Range (if available)
            'atr_ratio': RiskAnalyzer._calculate_atr_ratio(df),
            
            # Downside volatility (only negative returns)
            'downside_volatility': returns[returns < 0].std() * np.sqrt(252) if len(returns[returns < 0]) > 0 else 0
        }
        
        return metrics
    
    @staticmethod
    def _calculate_volatility_percentile(returns: pd.Series, window: int = 20) -> float:
        """Calculate where current volatility ranks historically."""
        if len(returns) < window * 2:
            return 50.0
        
        rolling_vol = returns.rolling(window).std()
        current_vol = rolling_vol.iloc[-1]
        percentile = stats.percentileofscore(rolling_vol.dropna(), current_vol)
        
        return percentile
    
    @staticmethod
    def _calculate_atr_ratio(df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range as percentage of price."""
        if not all(col in df.columns for col in ['high', 'low', 'close']):
            return 0.0
        
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        
        tr = pd.concat([
            high - low,
            (high - close).abs(),
            (low - close).abs()
        ], axis=1).max(axis=1)
        
        atr = tr.rolling(period).mean().iloc[-1]
        current_price = df['close'].iloc[-1]
        
        return (atr / current_price) * 100 if current_price > 0 else 0
    
    @staticmethod
    def calculate_var(
        df: pd.DataFrame,
        confidence_level: float = VAR_CONFIDENCE_LEVEL,
        investment: float = 10000
    ) -> Dict[str, float]:
        """
        Calculate Value at Risk (VaR).
        
        Args:
            df: DataFrame with returns
            confidence_level: Confidence level for VaR (e.g., 0.95)
            investment: Investment amount for VaR calculation
            
        Returns:
            Dictionary with VaR metrics
        """
        returns = df['daily_return'].dropna()
        
        # Historical VaR
        var_historical = np.percentile(returns, (1 - confidence_level) * 100)
        
        # Parametric VaR (assumes normal distribution)
        mean_return = returns.mean()
        std_return = returns.std()
        z_score = stats.norm.ppf(1 - confidence_level)
        var_parametric = mean_return + z_score * std_return
        
        # Conditional VaR (Expected Shortfall)
        cvar = returns[returns <= var_historical].mean()
        
        return {
            'var_1day_pct': abs(var_historical) * 100,
            'var_1day_amount': abs(var_historical) * investment,
            'var_parametric_pct': abs(var_parametric) * 100,
            'cvar_pct': abs(cvar) * 100 if not np.isnan(cvar) else 0,
            'cvar_amount': abs(cvar) * investment if not np.isnan(cvar) else 0,
            'confidence_level': confidence_level * 100
        }
    
    @staticmethod
    def calculate_trend_confidence(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate trend strength and confidence metrics.
        
        Args:
            df: DataFrame with price and indicator data
            
        Returns:
            Dictionary with trend metrics
        """
        close = df['close'].values
        
        # Linear regression for trend
        x = np.arange(len(close))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, close)
        
        # Trend direction
        if slope > 0:
            trend_direction = 'Bullish'
        elif slope < 0:
            trend_direction = 'Bearish'
        else:
            trend_direction = 'Neutral'
        
        # R-squared as trend strength
        r_squared = r_value ** 2
        
        # Trend strength interpretation
        if r_squared > 0.7:
            trend_strength = 'Strong'
        elif r_squared > 0.4:
            trend_strength = 'Moderate'
        else:
            trend_strength = 'Weak'
        
        # Short-term vs long-term trend alignment
        short_term_trend = 'Up' if df['close'].iloc[-1] > df['close'].iloc[-20] else 'Down'
        long_term_trend = 'Up' if df['close'].iloc[-1] > df['close'].iloc[-100] else 'Down' if len(df) > 100 else 'N/A'
        
        trend_alignment = 'Aligned' if short_term_trend == long_term_trend else 'Divergent'
        
        return {
            'trend_direction': trend_direction,
            'trend_strength': trend_strength,
            'r_squared': r_squared,
            'slope_per_day': slope,
            'p_value': p_value,
            'short_term_trend': short_term_trend,
            'long_term_trend': long_term_trend,
            'trend_alignment': trend_alignment,
            'statistically_significant': p_value < 0.05
        }
    
    @staticmethod
    def generate_risk_warnings(
        df: pd.DataFrame,
        volatility_metrics: Dict[str, float],
        var_metrics: Dict[str, float],
        trend_metrics: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """
        Generate risk warnings based on analysis.
        
        Returns:
            List of warning dictionaries with 'level' and 'message'
        """
        warnings = []
        
        # High volatility warning
        if volatility_metrics.get('volatility_percentile', 0) > 80:
            warnings.append({
                'level': 'high',
                'message': f"⚠️ High Volatility Alert: Current volatility is in the {volatility_metrics['volatility_percentile']:.0f}th percentile historically."
            })
        
        # Increasing volatility
        if volatility_metrics.get('recent_volatility', 0) > volatility_metrics.get('annualized_volatility', 0) * 1.5:
            warnings.append({
                'level': 'medium',
                'message': "📈 Recent volatility is significantly higher than historical average."
            })
        
        # High VaR warning
        if var_metrics.get('var_1day_pct', 0) > 3:
            warnings.append({
                'level': 'high',
                'message': f"💰 High Risk: 1-day VaR is {var_metrics['var_1day_pct']:.1f}% at {var_metrics['confidence_level']:.0f}% confidence."
            })
        
        # Weak trend warning
        if trend_metrics.get('trend_strength') == 'Weak':
            warnings.append({
                'level': 'low',
                'message': "📊 Weak Trend: Price movement lacks clear direction, increasing prediction uncertainty."
            })
        
        # Trend divergence warning
        if trend_metrics.get('trend_alignment') == 'Divergent':
            warnings.append({
                'level': 'medium',
                'message': "🔄 Trend Divergence: Short-term and long-term trends are not aligned."
            })
        
        # Drawdown warning
        df_with_dd = df.copy()
        df_with_dd['running_max'] = df_with_dd['close'].cummax()
        df_with_dd['drawdown'] = (df_with_dd['close'] - df_with_dd['running_max']) / df_with_dd['running_max']
        current_dd = df_with_dd['drawdown'].iloc[-1]
        
        if current_dd < -0.20:
            warnings.append({
                'level': 'high',
                'message': f"📉 Significant Drawdown: Stock is {abs(current_dd)*100:.1f}% below its peak."
            })
        
        return warnings
    
    @staticmethod
    def get_risk_summary(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive risk summary.
        
        Args:
            df: DataFrame with all data and indicators
            
        Returns:
            Complete risk analysis summary
        """
        volatility = RiskAnalyzer.calculate_volatility_metrics(df)
        var = RiskAnalyzer.calculate_var(df)
        trend = RiskAnalyzer.calculate_trend_confidence(df)
        warnings = RiskAnalyzer.generate_risk_warnings(df, volatility, var, trend)
        
        # Overall risk score (0-100)
        risk_score = RiskAnalyzer._calculate_risk_score(volatility, var, trend)
        
        return {
            'volatility_metrics': volatility,
            'var_metrics': var,
            'trend_metrics': trend,
            'warnings': warnings,
            'risk_score': risk_score,
            'risk_level': RiskAnalyzer._get_risk_level(risk_score)
        }
    
    @staticmethod
    def _calculate_risk_score(
        volatility: Dict[str, float],
        var: Dict[str, float],
        trend: Dict[str, Any]
    ) -> float:
        """Calculate overall risk score from 0-100."""
        score = 0
        
        # Volatility component (0-40 points)
        vol_percentile = volatility.get('volatility_percentile', 50)
        score += (vol_percentile / 100) * 40
        
        # VaR component (0-30 points)
        var_pct = min(var.get('var_1day_pct', 0), 10)  # Cap at 10%
        score += (var_pct / 10) * 30
        
        # Trend uncertainty component (0-30 points)
        r_squared = trend.get('r_squared', 0.5)
        trend_uncertainty = 1 - r_squared  # Higher uncertainty = higher risk
        score += trend_uncertainty * 30
        
        return min(score, 100)
    
    @staticmethod
    def _get_risk_level(score: float) -> str:
        """Convert risk score to risk level."""
        if score < 25:
            return 'Low'
        elif score < 50:
            return 'Moderate'
        elif score < 75:
            return 'High'
        else:
            return 'Very High'
