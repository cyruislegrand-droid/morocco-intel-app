import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import random

# Standard browser headers to prevent getting blocked by Moroccan firewalls
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
}

def extract_actors(text):
    """
    A lightweight Entity Extraction tool to find 'Hidden Links' without an API.
    Looks for specific Moroccan keywords in the text.
    """
    known_actors = {
        "Akhannouch": "Head of Gov",
        "BAM": "Bank Al-Maghrib",
        "Jouahri": "Bank Al-Maghrib",
        "OCP": "Phosphate Corp",
        "ONEE": "Energy/Water",
        "MASI": "Stock Market",
        "Drought": "Climate",
        "Protest": "Social Movement",
        "Attijariwafa": "Banking Sector"
    }
    
    found_actors = []
    for keyword, category in known_actors.items():
        if keyword.lower() in text.lower():
            found_actors.append(keyword)
            
    # Always return at least a generic actor to keep the Gephi network functioning
    return found_actors if found_actors else ["General Public", "Gov"]

def fetch_moroccan_news():
    """
    Scrapes major Moroccan news sources. Uses RSS feeds where possible 
    as they are cleaner and less likely to block basic requests.
    """
    articles = []
    
    # Target 1: Hespress English (RSS is easier to parse than their JS-heavy homepage)
    try:
        url = "https://en.hespress.com/feed"
        response = requests.get(url, headers=HEADERS, timeout=5)
        soup = BeautifulSoup(response.content, features="xml")
        
        items = soup.findAll('item')
        for item in items[:10]: # Grab top 10 articles
            title = item.title.text
            description = item.description.text
            
            articles.append({
                "title": title,
                "summary": description,
                "field": "Socio-Political",
                "actors": extract_actors(title + " " + description),
                "source": "Hespress"
            })
    except Exception as e:
        print(f"Hespress scrape failed: {e}")

    # Fallback / Simulated Data to guarantee the app renders the Network & Tables
    if len(articles) < 3:
        articles.extend([
            {
                "title": "Bank Al-Maghrib announces new interest rate policy",
                "summary": "Governor Jouahri signals a hold on rates amid inflation fears.",
                "field": "Economic",
                "actors": ["BAM", "Jouahri", "Banking Sector"],
                "source": "Simulated Intelligence"
            },
            {
                "title": "OCP reports record revenues despite global volatility",
                "summary": "The phosphate giant secures new African fertilizer deals.",
                "field": "Stocks",
                "actors": ["OCP", "Gov"],
                "source": "Simulated Intelligence"
            },
            {
                "title": "Agricultural sector faces severe drought warnings",
                "summary": "Farmers request Akhannouch government intervention.",
                "field": "Social",
                "actors": ["Akhannouch", "Drought"],
                "source": "Simulated Intelligence"
            }
        ])
        
    return pd.DataFrame(articles)

def fetch_masi_data():
    """
    Scraping live stock charts without a headless browser (Selenium) is highly brittle.
    For this 'no API' setup, we simulate a realistic Moroccan All Shares Index (MASI) 
    Random Walk to feed the predictive model.
    """
    # Simulate 30 days of MASI data centered around 13,000 points
    days = 30
    base_price = 13000
    
    # Generate a realistic random walk for stock prices
    volatility = 0.015 # 1.5% daily volatility
    returns = np.random.normal(loc=0.0005, scale=volatility, size=days)
    price_path = base_price * np.exp(np.cumsum(returns))
    
    dates = pd.date_range(end=pd.Timestamp.today(), periods=days)
    
    df = pd.DataFrame({
        'Date': dates,
        'Price': np.round(price_path, 2)
    })
    
    return df
