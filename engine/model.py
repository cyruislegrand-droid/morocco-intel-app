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
    
    # 2. Integrate Sentiment safely
    if not news_df.empty and 'sentiment_score' in news_df.columns:
        # Force numeric to prevent Pandas 3.0 TypeErrors
        sentiment_series = pd.to_numeric(news_df['sentiment_score'], errors='coerce')
        avg_sentiment = sentiment_series.mean()
        if pd.isna(avg_sentiment):
            avg_sentiment = 50.0
    else:
        avg_sentiment = 50.0  # Neutral baseline
        
    stock_df['sentiment_feature'] = float(avg_sentiment)
    
    # Drop NaNs created by Moving Averages
    data = stock_df.dropna().copy()
    return data

def train_masi_prediction(data):
    """
    Trains a model to predict the next day's closing price.
    """
    # Safety check
    if data is None or len(data) < 10:
        return None, 0
    
    # Features
    X = data[['MA5', 'MA20', 'sentiment_feature']].copy()
    
    # Target: Tomorrow's price (Shifts data up by 1)
    y = data['Price'].shift(-1)
    
    # THE FIX: Use .iloc for strict integer-location slicing.
    # We drop the last row because its target (y) is NaN. No .ffill() needed!
    X_train = X.iloc[:-1]
    y_train = y.iloc[:-1]
    
    # Train the Random Forest
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict "Tomorrow" using the very last row of features
    last_features = X.iloc[[-1]]
    
    # Force the prediction to be a standard Python float to prevent UI unpack errors
    prediction = float(model.predict(last_features)[0])
    
    # Calculate a mock confidence score based on market volatility
    mean_price = data['Price'].mean()
    mean_price = mean_price if mean_price != 0 else 1 # Prevent division by zero
    
    volatility_penalty = (data['Price'].std() / mean_price) * 1000
    confidence = max(10.0, min(95.0, 100.0 - float(volatility_penalty)))
    
    return round(prediction, 2), round(confidence, 1)

def get_market_outlook(prediction, current_price):
    """
    Returns a strategic recommendation based on the prediction delta.
    """
    if current_price == 0:
        return "NEUTRAL ↔️", "Market data unavailable."
        
    diff = ((prediction - current_price) / current_price) * 100
    
    if diff > 0.5:
        return "BULLISH 📈", "Technical momentum and macro sentiment point to upward growth."
    elif diff < -0.5:
        return "BEARISH 📉", "Downward pressure detected. Potential risk in socio-political landscape or price action."
    else:
        return "NEUTRAL ↔️", "Market is consolidating. No significant triggers detected."
