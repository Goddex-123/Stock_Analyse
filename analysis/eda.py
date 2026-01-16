"""
Exploratory Data Analysis Module
Provides visualization and statistical analysis functions
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, Dict, Any

from config import CHART_HEIGHT, CHART_TEMPLATE


class ExploratoryDataAnalysis:
    """
    Provides exploratory data analysis and visualization for stock data.
    """
    
    @staticmethod
    def create_price_chart(
        df: pd.DataFrame,
        symbol: str,
        show_volume: bool = True
    ) -> go.Figure:
        """
        Create an interactive candlestick chart with volume.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Stock symbol for title
            show_volume: Whether to show volume subplot
            
        Returns:
            Plotly figure object
        """
        if show_volume:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.7, 0.3],
                subplot_titles=(f'{symbol} Price', 'Volume')
            )
        else:
            fig = go.Figure()
        
        # Candlestick chart
        candlestick = go.Candlestick(
            x=df['date'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        )
        
        if show_volume:
            fig.add_trace(candlestick, row=1, col=1)
            
            # Volume bars
            colors = ['#26a69a' if close >= open else '#ef5350' 
                     for close, open in zip(df['close'], df['open'])]
            
            fig.add_trace(
                go.Bar(
                    x=df['date'],
                    y=df['volume'],
                    name='Volume',
                    marker_color=colors,
                    showlegend=False
                ),
                row=2, col=1
            )
        else:
            fig.add_trace(candlestick)
        
        fig.update_layout(
            title=f'{symbol} Stock Price',
            template=CHART_TEMPLATE,
            height=CHART_HEIGHT,
            xaxis_rangeslider_visible=False,
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def create_line_chart(
        df: pd.DataFrame,
        symbol: str,
        show_ma: bool = True
    ) -> go.Figure:
        """
        Create a line chart with moving averages.
        """
        fig = go.Figure()
        
        # Close price
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['close'],
            mode='lines',
            name='Close Price',
            line=dict(color='#2196f3', width=2)
        ))
        
        # Moving averages
        if show_ma:
            ma_columns = [col for col in df.columns if col.startswith('sma_')]
            colors = ['#ff9800', '#e91e63', '#9c27b0', '#4caf50', '#00bcd4']
            
            for i, col in enumerate(ma_columns[:5]):
                if col in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df['date'],
                        y=df[col],
                        mode='lines',
                        name=col.upper().replace('_', ' '),
                        line=dict(color=colors[i % len(colors)], width=1)
                    ))
        
        fig.update_layout(
            title=f'{symbol} Price with Moving Averages',
            yaxis_title='Price',
            template=CHART_TEMPLATE,
            height=CHART_HEIGHT,
            hovermode='x unified'
        )
        
        return fig
    
    @staticmethod
    def create_bollinger_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
        """
        Create Bollinger Bands visualization.
        """
        fig = go.Figure()
        
        # Upper band
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['bb_upper'],
            mode='lines',
            name='Upper Band',
            line=dict(color='rgba(255, 152, 0, 0.5)', width=1)
        ))
        
        # Lower band
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['bb_lower'],
            mode='lines',
            name='Lower Band',
            line=dict(color='rgba(255, 152, 0, 0.5)', width=1),
            fill='tonexty',
            fillcolor='rgba(255, 152, 0, 0.1)'
        ))
        
        # Middle band (SMA)
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['bb_middle'],
            mode='lines',
            name='Middle Band (SMA)',
            line=dict(color='#ff9800', width=1, dash='dash')
        ))
        
        # Close price
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['close'],
            mode='lines',
            name='Close Price',
            line=dict(color='#2196f3', width=2)
        ))
        
        fig.update_layout(
            title=f'{symbol} Bollinger Bands',
            yaxis_title='Price',
            template=CHART_TEMPLATE,
            height=CHART_HEIGHT,
            hovermode='x unified'
        )
        
        return fig
    
    @staticmethod
    def create_rsi_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
        """
        Create RSI indicator chart.
        """
        fig = go.Figure()
        
        # RSI line
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['rsi'],
            mode='lines',
            name='RSI',
            line=dict(color='#9c27b0', width=2)
        ))
        
        # Overbought line
        fig.add_hline(y=70, line_dash="dash", line_color="red", 
                     annotation_text="Overbought (70)")
        
        # Oversold line
        fig.add_hline(y=30, line_dash="dash", line_color="green",
                     annotation_text="Oversold (30)")
        
        # Neutral line
        fig.add_hline(y=50, line_dash="dot", line_color="gray")
        
        fig.update_layout(
            title=f'{symbol} Relative Strength Index (RSI)',
            yaxis_title='RSI',
            yaxis=dict(range=[0, 100]),
            template=CHART_TEMPLATE,
            height=300,
            hovermode='x unified'
        )
        
        return fig
    
    @staticmethod
    def create_macd_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
        """
        Create MACD indicator chart.
        """
        fig = make_subplots(rows=1, cols=1)
        
        # MACD line
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['macd'],
            mode='lines',
            name='MACD',
            line=dict(color='#2196f3', width=2)
        ))
        
        # Signal line
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['macd_signal'],
            mode='lines',
            name='Signal',
            line=dict(color='#ff9800', width=2)
        ))
        
        # Histogram
        colors = ['#26a69a' if val >= 0 else '#ef5350' 
                 for val in df['macd_histogram']]
        
        fig.add_trace(go.Bar(
            x=df['date'],
            y=df['macd_histogram'],
            name='Histogram',
            marker_color=colors
        ))
        
        fig.update_layout(
            title=f'{symbol} MACD',
            yaxis_title='MACD',
            template=CHART_TEMPLATE,
            height=300,
            hovermode='x unified'
        )
        
        return fig
    
    @staticmethod
    def create_returns_distribution(df: pd.DataFrame, symbol: str) -> go.Figure:
        """
        Create returns distribution histogram.
        """
        returns = df['daily_return'].dropna()
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=returns,
            nbinsx=50,
            name='Daily Returns',
            marker_color='#2196f3',
            opacity=0.7
        ))
        
        # Add normal distribution overlay
        mean_ret = returns.mean()
        std_ret = returns.std()
        
        x_range = np.linspace(returns.min(), returns.max(), 100)
        normal_dist = (1 / (std_ret * np.sqrt(2 * np.pi))) * \
                      np.exp(-0.5 * ((x_range - mean_ret) / std_ret) ** 2)
        
        # Scale to match histogram
        scale_factor = len(returns) * (returns.max() - returns.min()) / 50
        
        fig.add_trace(go.Scatter(
            x=x_range,
            y=normal_dist * scale_factor,
            mode='lines',
            name='Normal Distribution',
            line=dict(color='#ff9800', width=2)
        ))
        
        fig.update_layout(
            title=f'{symbol} Daily Returns Distribution',
            xaxis_title='Daily Return',
            yaxis_title='Frequency',
            template=CHART_TEMPLATE,
            height=400,
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def calculate_drawdown(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate drawdown series.
        """
        df = df.copy()
        
        # Running maximum
        df['running_max'] = df['close'].cummax()
        
        # Drawdown
        df['drawdown'] = (df['close'] - df['running_max']) / df['running_max']
        
        return df
    
    @staticmethod
    def create_drawdown_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
        """
        Create drawdown visualization.
        """
        df = ExploratoryDataAnalysis.calculate_drawdown(df)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['drawdown'] * 100,
            mode='lines',
            name='Drawdown',
            fill='tozeroy',
            line=dict(color='#ef5350', width=1),
            fillcolor='rgba(239, 83, 80, 0.3)'
        ))
        
        # Maximum drawdown
        max_dd = df['drawdown'].min() * 100
        max_dd_date = df.loc[df['drawdown'].idxmin(), 'date']
        
        fig.add_annotation(
            x=max_dd_date,
            y=max_dd,
            text=f'Max Drawdown: {max_dd:.1f}%',
            showarrow=True,
            arrowhead=2
        )
        
        fig.update_layout(
            title=f'{symbol} Drawdown Analysis',
            yaxis_title='Drawdown (%)',
            template=CHART_TEMPLATE,
            height=350,
            hovermode='x unified'
        )
        
        return fig
    
    @staticmethod
    def get_summary_statistics(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate summary statistics for the stock.
        """
        returns = df['daily_return'].dropna()
        
        stats = {
            'start_date': df['date'].min().strftime('%Y-%m-%d'),
            'end_date': df['date'].max().strftime('%Y-%m-%d'),
            'trading_days': len(df),
            'current_price': df['close'].iloc[-1],
            'start_price': df['close'].iloc[0],
            'total_return': ((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100,
            'mean_daily_return': returns.mean() * 100,
            'std_daily_return': returns.std() * 100,
            'annualized_return': returns.mean() * 252 * 100,
            'annualized_volatility': returns.std() * np.sqrt(252) * 100,
            'sharpe_ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0,
            'max_price': df['high'].max(),
            'min_price': df['low'].min(),
            'max_drawdown': ExploratoryDataAnalysis.calculate_drawdown(df)['drawdown'].min() * 100,
            'positive_days': (returns > 0).sum(),
            'negative_days': (returns < 0).sum(),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis()
        }
        
        return stats
