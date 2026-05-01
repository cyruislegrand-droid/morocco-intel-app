import pandas as pd
import networkx as nx
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize VADER
analyzer = SentimentIntensityAnalyzer()

def compute_sentiment(text):
    """
    Calculates a sentiment score for the given text.
    """
    if not text or pd.isna(text):
        return 0.0
    try:
        # Hybrid score
        blob_score = TextBlob(str(text)).sentiment.polarity
        vader_score = analyzer.polarity_scores(str(text))['compound']
        return (blob_score + vader_score) / 2
    except Exception:
        return 0.0

def build_actor_network(news_df):
    """
    Creates a structural network of actors.
    """
    G = nx.Graph()
    if news_df.empty:
        return G
        
    for _, row in news_df.iterrows():
        actors = row.get('actors', [])
        if isinstance(actors, list):
            for i in range(len(actors)):
                for j in range(i + 1, len(actors)):
                    if G.has_edge(actors[i], actors[j]):
                        G[actors[i]][actors[j]]['weight'] += 1
                    else:
                        G.add_edge(actors[i], actors[j], weight=1)
    return G
