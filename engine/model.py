import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

def prepare_predictive_data(stock_df, news_df):
    """
    Merges stock prices with sentiment data to create a feature set.
    """
    # 1. Feature Engineering: Moving Averages
    stock_df['MA5'] = stock_df['Price'].rolling(window=5).mean()
    stock_df['MA20'] = stock_df['Price'].rolling(window=20).mean()
    
    # 2. Integrate Sentiment (Simplified logic)
    # We assume news_df has a 'sentiment_score' column from analysis.py
    avg_sentiment = news_df['sentiment_score'].mean() if not news_df.empty else 0
    stock_df['sentiment_feature'] = avg_sentiment
    
    # Drop NaNs created by Moving Averages
    data = stock_df.dropna()
    return data

def train_masi_prediction(data):
    """
    Trains a model to predict the next day's closing price.
    """
    if len(data) < 10:
        return None, 0
    
    # Features: MA5, MA20, Sentiment
    X = data[['MA5', 'MA20', 'sentiment_feature']]
    y = data['Price'].shift(-1).fillna(method='ffill') # Target is next price
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Predict "Tomorrow" using the last row of data
    last_features = X.tail(1)
    prediction = model.predict(last_features)[0]
    
    # Calculate a mock confidence score based on sentiment volatility
    confidence = 100 - (data['sentiment_feature'].std() * 100)
    
    return round(prediction, 2), round(confidence, 1)

def get_market_outlook(prediction, current_price):
    """
    Returns a strategic recommendation.
    """
    diff = ((prediction - current_price) / current_price) * 100
    if diff > 0.5:
        return "BULLISH 📈", "Strong social/economic sentiment suggests market growth."
    elif diff < -0.5:
        return "BEARISH 📉", "Political tension or negative economic indicators detected."
    else:
        return "NEUTRAL ↔️", "Market consolidating; no significant triggers found."
