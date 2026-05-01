import requests
from bs4 import BeautifulSoup

def fetch_moroccan_news():
    # Example: Scraping L'Economiste or Hespress
    # In a real app, you'd iterate through multiple RSS feeds or HTML structures
    data = [
        {"title": "New Investment Charter Impact", "field": "Economic", "summary": "...", "actors": ["Gov", "Private Sector"]},
        {"title": "MASI Fluctuations", "field": "Stocks", "summary": "...", "actors": ["OCP", "Attijari"]}
    ]
    return pd.DataFrame(data)

def fetch_masi_data():
    # Scrape 'https://www.casablanca-bourse.com/'
    # Return a dataframe of recent prices
    return pd.DataFrame({'Price': np.random.randn(20).cumsum()})