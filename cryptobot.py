import ccxt
import numpy as np
import pandas as pd
import time
import json
import os
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Charger les variables d'environnement
load_dotenv()
api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")

# Configuration API Binance
exchange = ccxt.binance({
    'apiKey': api_key,
    'secret': api_secret,
    'options': {'defaultType': 'future'}
})

symbol = 'BTC/USDT'
timeframe = '15m'
investment = 100  # Montant en USDT par trade

# Chemin du fichier de statut
status_file = "/var/www/html/cryptobot/status.json"

def update_status(action):
    status = {
        "last_update": time.strftime("%Y-%m-%d %H:%M:%S"),
        "last_action": action
    }
    with open(status_file, "w") as f:
        json.dump(status, f)

# Chargement des données de marché
def fetch_data():
    bars = exchange.fetch_ohlcv(symbol, timeframe, limit=200)
    df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(window=10).std()
    df['momentum'] = df['close'].diff()
    df.dropna(inplace=True)
    return df

# Feature Engineering + IA
rf_model = RandomForestRegressor(n_estimators=200)
scaler = StandardScaler()

def train_model(df):
    df.dropna(inplace=True)
    features = ['open', 'high', 'low', 'close', 'volume', 'volatility', 'momentum']
    X = df[features].values
    y = df['returns'].shift(-1).fillna(0).values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    rf_model.fit(X_train, y_train)

def predict_market(df):
    features = ['open', 'high', 'low', 'close', 'volume', 'volatility', 'momentum']
    X = df[features].values[-1].reshape(1, -1)
    X = scaler.transform(X)
    return rf_model.predict(X)[0]

# Trading Logic
def execute_trade():
    df = fetch_data()
    train_model(df)
    prediction = predict_market(df)
    balance = exchange.fetch_balance()
    usdt_available = balance['total']['USDT']
    action = "Aucune action"
    
    if prediction > 0 and usdt_available >= investment:
        action = "Achat de BTC"
        order = exchange.create_market_buy_order(symbol, investment / df['close'].iloc[-1])
    elif prediction < 0:
        btc_available = balance['total']['BTC']
        if btc_available > 0:
            action = "Vente de BTC"
            order = exchange.create_market_sell_order(symbol, btc_available)
    
    update_status(action)

# Initialisation du statut
update_status("Bot démarré")

# Boucle principale
while True:
    try:
        execute_trade()
        time.sleep(900)  # Attendre 15 minutes entre chaque trade
    except Exception as e:
        update_status(f"Erreur: {str(e)}")
        time.sleep(900)
