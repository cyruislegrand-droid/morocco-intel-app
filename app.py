import os
import sys

# 1. Path injection to ensure 'engine' is found on Streamlit Cloud
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components

# 2. Engine imports
from engine.scraper import fetch_moroccan_news, fetch_masi_data
from engine.analysis import compute_sentiment, build_actor_network
from engine.model import prepare_predictive_data, train_masi_prediction, get_market_outlook

# 3. Page Configuration
st.set_page_config(page_title="Morocco Intel Engine", page_icon="🇲🇦", layout="wide")

st.title("🇲🇦 Morocco Real-Time Strategic Intelligence")
st.markdown("### Economical, Political, and Social Decision-Support System")

# --- SIDEBAR: Controls ---
st.sidebar.header("Control Panel")
target_field = st.sidebar.selectbox("Focus Area", ["Stock Market (MASI)", "Social Stability", "Political Landscape"])
refresh = st.sidebar.button("Fetch Live Data")

# --- DATA INGESTION ---
with st.spinner("Analyzing Moroccan Sources (Hespress, L'Economiste, BAM)..."):
    news_df = fetch_moroccan_news()
    stock_df = fetch_masi_data()

# Compute sentiment and stability score (this also adds 'sentiment_score' to news_df)
risk_val = compute_sentiment(news_df) 

# --- SECTION 1: THE NETWORK (Gephi-style Hidden Links) ---
st.header("🔗 Hidden Actor & Event Mapping")
st.markdown("Visualizing co-occurrences of key figures, institutions, and events in the Moroccan news cycle.")

G = build_actor_network(news_df)
if G.number_of_nodes() > 0:
    # Build PyVis Network
    net = Network(height="500px", width="100%", bgcolor="#222222", font_color="white")
    net.from_nx(G)
    net.save_graph("network.html")
    
    # Read and render the HTML
    with open("network.html", 'r', encoding='utf-8') as f:
        html_string = f.read()
    components.html(html_string, height=550)
else:
    st.info("Insufficient data to build the actor network. Try refreshing the feed.")

# --- SECTION 2: PREDICTIVE MODELING & RISK ---
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Stock Market Prediction (MASI)")
    st.line_chart(stock_df.set_index('Date')['Price'])
    
    # Process data for the ML model
    if 'sentiment_score' not in news_df.columns:
        news_df['sentiment_score'] = 0.5 # Fallback if sentiment failed
        
    ml_data = prepare_predictive_data(stock_df, news_df)
    prediction, confidence = train_masi_prediction(ml_data)
    
    if prediction:
        current_price = stock_df['Price'].iloc[-1]
        outlook, reason = get_market_outlook(prediction, current_price)
        
        # Visualizing the Decision Tool
        st.metric(
            label="Predicted Next Closing", 
            value=f"{prediction} MAD", 
            delta=f"{round(prediction - current_price, 2)} MAD"
        )
        st.write(f"**Strategy:** {outlook}")
        st.caption(f"**Reasoning:** {reason}")
        st.progress(int(confidence), text=f"Model Confidence: {int(confidence)}%")
    else:
        st.warning("Insufficient historical data to run predictive model.")

with col2:
    st.subheader("⚖️ Social & Political Stability Index")
    st.markdown("Analyzes the sentiment of current events to project socio-political risk.")
    
    # Native Streamlit Stability Indicator (Replaced the faulty st.gauge)
    delta_text = "Positive Trend" if risk_val >= 50 else "- High Risk Alert"
    delta_color = "normal" if risk_val >= 50 else "inverse"
    
    st.metric(
        label="Current Stability Score", 
        value=f"{risk_val} / 100", 
        delta=delta_text, 
        delta_color=delta_color
    )

    try:
        st.progress(int(risk_val), text="0 = High Risk | 100 = High Stability")
    except ValueError:
        st.progress(50, text="Awaiting sufficient data...")
        
    st.info("💡 **Insight:** This index bridges linguistic gaps by analyzing both text structures to detect hidden market panic or optimism before it hits traditional channels.")

# --- SECTION 3: STRATEGIC INTELLIGENCE ZOOM ---
st.markdown("---")
st.header("🔍 Strategic Intelligence Zoom")
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("🕵️ Hidden Actor Clusters")
    # Dynamically find the most common actor in the news cycle
    if not news_df.empty and 'actors' in news_df.columns:
        all_actors = [actor for actors_list in news_df['actors'] if isinstance(actors_list, list) for actor in actors_list]
        if all_actors:
            top_actor = pd.Series(all_actors).mode()[0]
            st.write(f"Identified anomalous high occurrence of **{top_actor}** connected to recent policy/economic shifts.")
        else:
            st.write("Gathering actor data...")
    else:
        st.write("Gathering actor data...")

with col_b:
    st.subheader("💡 Market Curiosities")
    if risk_val < 45:
        st.warning("Anomaly Detected: Social sentiment is diverging downwards. Potential correction risk for MASI.")
    elif risk_val > 60:
        st.success("Macro alignment: Positive policy sentiment supports continued market growth.")
    else:
        st.info("Market is consolidating. No immediate macro-anomalies detected.")

# --- SECTION 4: RAW INTELLIGENCE FEED ---
st.markdown("---")
st.header("📰 Raw Intelligence Feed")
if not news_df.empty:
    for index, row in news_df.head(10).iterrows():
        # Fallbacks added via .get() so missing scrape data never crashes the app
        title = row.get('title', 'No Title')
        field = row.get('field', 'General')
        with st.expander(f"{title} - {field}"):
            st.write(row.get('summary', 'No summary available.'))
            actors = row.get('actors', [])
            st.caption(f"**Hidden Actors identified:** {', '.join(actors) if isinstance(actors, list) else 'None'}")
            st.caption(f"**Source:** {row.get('source', 'Unknown')} | **Sentiment Impact:** {round(row.get('sentiment_score', 0), 2)}")
else:
    st.write("No intelligence data gathered. Please check the scrapers.")
