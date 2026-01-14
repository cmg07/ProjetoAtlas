import yfinance as yf
import pandas as pd
import numpy as np
from textblob import TextBlob

class DataIngestionISA:
    def __init__(self, ticker):
        self.ticker = ticker
        self.stock = yf.Ticker(ticker)

    def fetch_all(self):
        # OHLCV e Fundamentais (Camada de Entrada)
        hist = self.stock.history(period="1y")
        info = self.stock.info
        
        # News Sentiment via NLP (News ISA)
        news = self.stock.news
        sentiments = []
        
        for n in news:
            # Uso do .get() evita o KeyError se 'title' não existir
            title = n.get('title')
            if title:
                analysis = TextBlob(title).sentiment.polarity
                sentiments.append(analysis)
        
        avg_sentiment = np.mean(sentiments) if sentiments else 0
        
        return {
            "history": hist,
            "sentiment": avg_sentiment,
            "info": info
        }