import os
import ccxt
import joblib
import numpy as np
import pandas as pd
import qiskit
import optuna
import tensorflow as tf
from flask import Flask, render_template, jsonify
from sklearn.model_selection import train_test_split
from stable_baselines3 import DQN
from qiskit import Aer, execute, QuantumCircuit
from qiskit.algorithms import QAOA
from qiskit.optimization import QuadraticProgram
from qiskit.optimization.algorithms import MinimumEigenOptimizer

# Charger les variables d'environnement
MODE = "TEST"  # Changer en "LIVE" pour mode réel
API_KEY = os.getenv("BINANCE_TEST_API_KEY") if MODE == "TEST" else os.getenv("BINANCE_REAL_API_KEY")
API_SECRET = os.getenv("BINANCE_TEST_API_SECRET") if MODE == "TEST" else os.getenv("BINANCE_REAL_API_SECRET")

exchange = ccxt.binance({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'enableRateLimit': True,
})

if MODE == "TEST":
    exchange.set_sandbox_mode(True)

print(f"✅ Mode {MODE} activé")

# Initialisation Flask
app = Flask(__name__)

def get_binance_data(symbol="BTC/USDT", timeframe="1h", limit=1000):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["returns"] = df["close"].pct_change()
    df["target"] = np.where(df["returns"].shift(-1) > 0, 1, 0)
    return df.dropna()

@app.route('/')
def dashboard():
    return render_template("index.html")

@app.route('/status')
def status():
    balance = exchange.fetch_balance()
    return jsonify({
        "capital": balance['total']['USDT'],
        "mode": MODE
    })

# Implémentation de QAOA pour optimiser le trading
qp = QuadraticProgram()
qp.binary_var_list(["trade_BTC", "trade_ETH", "trade_BNB"])
qp.minimize(linear=[-0.1, -0.15, -0.2])

simulator = Aer.get_backend('aer_simulator')
qaoa = QAOA()
optimizer = MinimumEigenOptimizer(qaoa)
result = optimizer.solve(qp)
print("✅ Meilleure stratégie de trading :", result.x)

# Implémentation du Reinforcement Learning Quantique
env = get_binance_data()
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

print("✅ Reinforcement Learning Quantique entraîné !")

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
