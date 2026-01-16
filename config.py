"""
Configuration settings for Stock Analyzer System
"""

# Data Settings
DEFAULT_YEARS_HISTORY = 5
DEFAULT_PREDICTION_DAYS = 30

# Technical Indicator Settings
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2

# Rolling Statistics Settings
ROLLING_WINDOWS = [5, 10, 20, 50, 200]

# Model Settings
TRAIN_TEST_SPLIT = 0.8
RANDOM_STATE = 42

# XGBoost Parameters
XGBOOST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_STATE
}

# LSTM Parameters
LSTM_SEQUENCE_LENGTH = 60
LSTM_EPOCHS = 50
LSTM_BATCH_SIZE = 32
LSTM_UNITS = 50

# Risk Analysis Settings
VAR_CONFIDENCE_LEVEL = 0.95
VOLATILITY_WINDOW = 20

# UI Settings
CHART_HEIGHT = 500
CHART_TEMPLATE = "plotly_dark"

# Disclaimer Text
DISCLAIMER = """
⚠️ **IMPORTANT DISCLAIMER**

This application is for **educational and demonstration purposes only**.

- **NOT FINANCIAL ADVICE**: Predictions are probabilistic estimates and should NOT be used as the sole basis for investment decisions.
- **NO GUARANTEES**: Past performance does not guarantee future results. Stock markets are inherently unpredictable.
- **EXTERNAL FACTORS**: This model cannot account for news events, policy changes, market sentiment, or other external factors.
- **CONSULT PROFESSIONALS**: Always consult a licensed financial advisor before making investment decisions.

By using this application, you acknowledge that you understand these limitations.
"""
