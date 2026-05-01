import pandas as pd
import networkx as nx
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize the VADER sentiment analyzer once to save resources
analyzer = SentimentIntensityAnalyzer()

def compute_sentiment(input_data):
    """
    Calculates sentiment and stability score. 
    Gracefully handles both individual text strings and pandas DataFrames.
    """
    # --- 1. Handling a DataFrame Input (From app.py) ---
    if isinstance(input_data, pd.DataFrame):
        if input_data.empty:
            return 50.0  # Default neutral stability score

        # Helper function to score a single row's text
        def score_text(text):
            if not text or pd.isna(text):
                return 0.0
            try:
                blob = TextBlob(str(text)).sentiment.polarity
                vader = analyzer.polarity_scores(str(text))['compound']
                return (blob + vader) / 2
            except Exception:
                return 0.0

        # Apply the score to the 'summary' column to power the ML model later
        text_column = 'summary' if 'summary' in input_data.columns else 'title'
        input_data['sentiment_score'] = input_data[text_column].apply(score_text)

        # Calculate a "Stability Score" (0 to 100 scale) for the UI Gauge
        # Positive sentiment = High Stability (close to 100)
        # Negative sentiment = Low Stability (close to 0)
        avg_sentiment = input_data['sentiment_score'].mean()
        stability_score = 50 + (avg_sentiment * 50) 
        
        # Ensure it stays strictly within 0-100 bounds
        return round(max(0.0, min(100.0, stability_score)), 1)

    # --- 2. Handling a Single String Input ---
    if not input_data or pd.isna(input_data):
        return 0.0
        
    try:
        blob_score = TextBlob(str(input_data)).sentiment.polarity
        vader_score = analyzer.polarity_scores(str(input_data))['compound']
        return (blob_score + vader_score) / 2
    except Exception:
        return 0.0


def build_actor_network(news_df):
    """
    Builds a NetworkX graph showing hidden links between entities.
    Entities that appear in the same article are linked.
    """
    G = nx.Graph()
    
    # If the dataframe is empty or missing data, return an empty graph to prevent crashes
    if news_df is None or news_df.empty or 'actors' not in news_df.columns:
        return G
        
    for _, row in news_df.iterrows():
        actors = row.get('actors', [])
        
        # Ensure actors is a list and has at least two items to form a hidden link
        if isinstance(actors, list) and len(actors) > 1:
            for i in range(len(actors)):
                for j in range(i + 1, len(actors)):
                    node1, node2 = actors[i], actors[j]
                    
                    # Add or update the weight of the connection
                    if G.has_edge(node1, node2):
                        G[node1][node2]['weight'] += 1
                    else:
                        G.add_edge(node1, node2, weight=1)
                        
        # Ensure solitary actors are still added as dots on the map
        elif isinstance(actors, list) and len(actors) == 1:
            if not G.has_node(actors[0]):
                G.add_node(actors[0])
                
    return G
