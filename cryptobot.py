import ccxt
import numpy as np
import pandas as pd
import time
import json
import talib
from sklearn.ensemble import RandomForestRegressor

# Configuration API Binance
api_key = "YOUR_API_KEY"
api_secret = "YOUR_API_SECRET"
exchange = ccxt.binance({
    'apiKey': api_key,
    'secret': api_secret,
    'options': {'defaultType': 'future'}
})

symbol = 'BTC/USDT'
timeframe = '15m'
investment = 100  # Montant en USDT par trade

# Chargement des données de marché
def fetch_data():
    bars = exchange.fetch_ohlcv(symbol, timeframe, limit=100)
    df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['returns'] = df['close'].pct_change()
    return df

# Feature Engineering + IA
rf_model = RandomForestRegressor(n_estimators=100)

def train_model(df):
    df.dropna(inplace=True)
    X = df[['open', 'high', 'low', 'close', 'volume']].values
    y = df['returns'].shift(-1).fillna(0).values
    rf_model.fit(X, y)

def predict_market(df):
    X = df[['open', 'high', 'low', 'close', 'volume']].values[-1].reshape(1, -1)
    return rf_model.predict(X)[0]

# Trading Logic
def execute_trade():
    df = fetch_data()
    train_model(df)
    prediction = predict_market(df)
    balance = exchange.fetch_balance()
    usdt_available = balance['total']['USDT']
    
    if prediction > 0 and usdt_available >= investment:
        print("Achat de BTC")
        order = exchange.create_market_buy_order(symbol, investment / df['close'].iloc[-1])
        return order
    elif prediction < 0:
        print("Vente de BTC")
        btc_available = balance['total']['BTC']
        if btc_available > 0:
            order = exchange.create_market_sell_order(symbol, btc_available)
            return order
    return None

# Boucle principale
while True:
    try:
        execute_trade()
        time.sleep(900)  # Attendre 15 minutes entre chaque trade
    except Exception as e:
        print(f"Erreur: {e}")
