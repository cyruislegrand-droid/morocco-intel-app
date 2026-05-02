import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def prepare_predictive_data(stock_df, news_df):
    """
    Merges stock prices with sentiment data to create a feature set.
    """
    # 1. Feature Engineering: Moving Averages
    stock_df['MA5'] = stock_df['Price'].rolling(window=5).mean()
    stock_df['MA20'] = stock_df['Price'].rolling(window=20).mean()
    
    # 2. Integrate Sentiment
    if not news_df.empty and 'sentiment_score' in news_df.columns:
        avg_sentiment = news_df['sentiment_score'].mean()
    else:
        avg_sentiment = 0.5  # Neutral baseline
        
    stock_df['sentiment_feature'] = avg_sentiment
    
    # Drop NaNs created by Moving Averages so the ML model doesn't crash
    data = stock_df.dropna().copy()
    return data

def train_masi_prediction(data):
    """
    Trains a model to predict the next day's closing price.
    """
    if len(data) < 10:
        return None, 0
    
    # Features: MA5, MA20, Sentiment
    X = data[['MA5', 'MA20', 'sentiment_feature']]
    
    # THE FIX: Pandas 3.0 requires .ffill() instead of .fillna(method='ffill')
    y = data['Price'].shift(-1).ffill()
    
    # We drop the very last row for training because its 'Target' (tomorrow) doesn't exist yet
    X_train = X[:-1]
    y_train = y[:-1]
    
    # Train the Random Forest
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict "Tomorrow" using the very last row of data
    last_features = X.tail(1)
    prediction = model.predict(last_features)[0]
    
    # Calculate a mock confidence score based on market volatility
    volatility_penalty = (data['Price'].std() / data['Price'].mean()) * 1000
    confidence = max(10, min(95, 100 - volatility_penalty)) # Bound between 10% and 95%
    
    return round(prediction, 2), round(confidence, 1)

def get_market_outlook(prediction, current_price):
    """
    Returns a strategic recommendation based on the prediction delta.
    """
    diff = ((prediction - current_price) / current_price) * 100
    
    if diff > 0.5:
        return "BULLISH 📈", "Technical momentum and macro sentiment point to upward growth."
    elif diff < -0.5:
        return "BEARISH 📉", "Downward pressure detected. Potential risk in socio-political landscape or price action."
    else:
        return "NEUTRAL ↔️", "Market is consolidating. No significant triggers detected."
