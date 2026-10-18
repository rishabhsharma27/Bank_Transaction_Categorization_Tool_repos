import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.models import Sequential
from pathlib import Path



def train_model():
    # 1. Generate or Load Data (Replace with pd.read_csv('your_stock_data.csv'))
    # Using synthetic sine wave + trend as a proxy for cyclical price action
    """ 

    dates = pd.date_range(start='2023-01-01', periods=500, freq='D')
    values = (
        np.sin(np.linspace(0, 50, 500))
        + np.linspace(0, 5, 500)
        + np.random.normal(0, 0.1, 500)
    )
    df = pd.DataFrame({'Date': dates, 'Close': values}) 
    """

    # Set up paths
    current_dir = Path(__file__).resolve().parent
    root_dir = current_dir.parent
    data_path = root_dir / "data"
    models_dir = root_dir / "models"
    models_dir.mkdir(exist_ok=True)

    # 1. Generate or Load Data (Replace with pd.read_csv('your_stock_data.csv'))
    df = pd.read_csv('data/sp500_stocks_2025.csv')  # Ensure your CSV has 'Date' and 'Close' columns 

    # 2. Preprocess and Scale Data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['close']])

    # 3. Create Time-Series Sequences (Lookback Window)
    # LSTMs require 3D input: [samples, time_steps, features]

    LOOK_BACK = 60
    X, y = create_sequences(scaled_data, LOOK_BACK)


    # Reshape X to be [samples, time_steps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Split into Train and Test sets (80/20 split)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
"""
    # 4. Build the LSTM Architecture
    model = Sequential([
        Input(shape=(X_train.shape[1], 1)),
        LSTM(5, return_sequences=True),
        LSTM(5, return_sequences=False),
        Dense(5),
        Dense(1),
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()

    # 5. Train the Model
    history = model.fit(
        X_train,
        y_train,
        batch_size=32,
        epochs=1,
        validation_data=(X_test, y_test),
        verbose=1,
    )

    # 6. Save Model and Feature Schema Artifacts for Production Inference
    model_path = models_dir / "lstm_stock_model.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)


    print(f"📁 Model saved to: {model_path}")
"""

    model_path = models_dir / "lstm_stock_model.pkl"

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # 7. Make Predictions & Inverse Scale
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    # 8. Plot Results
    plt.figure(figsize=(12, 6))
    plt.plot(df['date'].iloc[-len(y_test) :], y_test_scaled, label='Actual Price')
    plt.plot(
        df['date'].iloc[-len(y_test) :], predictions, label='LSTM Predicted Price'
    )
    plt.title('Stock Trend Prediction with LSTM')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    plt.savefig(root_dir / "reports" / 'lstm_stock_prediction.png', dpi=300, bbox_inches="tight")

def create_sequences(dataset, look_back=60):
    X, y = [], []
    for i in range(look_back, len(dataset)):
        X.append(dataset[i - look_back : i, 0])
        y.append(dataset[i, 0])
        
    return np.array(X), np.array(y)


if __name__ == "__main__":
    train_model()