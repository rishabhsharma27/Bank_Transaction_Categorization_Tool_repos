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


    # 3. Create Time-Series Sequences (Lookback Window)
    # LSTMs require 3D input: [samples, time_steps, features]

    LOOK_BACK = 60
    X, y, meta_df, scalers_dict = create_grouped_sequences_with_metadata(df, LOOK_BACK)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Split into Train and Test sets (80/20 split)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]


    # 4. Build the LSTM Architecture
    model = Sequential([
        Input(shape=(X_train.shape[1], 1)),
        LSTM(50, return_sequences=True),
        LSTM(50, return_sequences=False),
        Dense(125),
        Dense(1),
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()

    # 5. Train the Model
    history = model.fit(
        X_train,
        y_train,
        batch_size=32,
        epochs=20,
        validation_data=(X_test, y_test),
        verbose=1,
    )

    # 6. Save Model and Feature Schema Artifacts for Production Inference
    model_path = models_dir / "lstm_stock_model.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)


    print(f"📁 Model saved to: {model_path}")
    
    

    model_path = models_dir / "lstm_stock_model.pkl"
  
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # 7. Make Predictions & Inverse Scale

    predictions = model.predict(X_test)

    inverse_transform_predictions(predictions, scalers_dict, meta_df, y_test)



def create_grouped_sequences_with_metadata(df, look_back=60):
    X_all , y_all = [], []  
    metadata = []
    scalers = {}  # 1. Initialize an empty dictionary to store scalers
    count = round(df['symbol'].count()/df['symbol'].nunique())
    
    for ticker, group in df.groupby('symbol'):
        # Sort just to be safe on chronological order
        group = group.sort_values('date')
        prices = group[['close']].values
        dates = group['date'].values
        if len(prices) !=count:
            continue
        if len(prices) > look_back:
        # 2. Create and fit a unique scaler for THIS specific company
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_prices = scaler.fit_transform(prices)

        # 3. Store the fitted scaler in the dictionary using the ticker name
            scalers[ticker] = scaler

        # Create the sliding windows
            for i in range(look_back, len(scaled_prices)):
                X_all.append(scaled_prices[i - look_back : i, 0])
                y_all.append(scaled_prices[i, 0])
                metadata.append({'Ticker': ticker, 'Date': dates[i]})

    return (
      np.array(X_all),
      np.array(y_all),
      pd.DataFrame(metadata),
      scalers,)


    
def inverse_transform_predictions(predictions, scalers_dict, metadata, y_test):
    # 1. Align test metadata with your predictions and actuals
    test_meta_df = metadata.iloc[-len(predictions) :].copy()
    test_meta_df['Predicted_Price'] = predictions.flatten()
    test_meta_df['Actual_Price'] = y_test.flatten()

    # 2. Inverse transform all companies using their respective scalers
    all_inverse_transformed = []

    for ticker, group in test_meta_df.groupby('Ticker'):
        if ticker in scalers_dict:
            scaler = scalers_dict[ticker]

            # Reshape to 2D for MinMaxScaler inverse_transform
            preds_2d = group['Predicted_Price'].values.reshape(-1, 1)
            actuals_2d = group['Actual_Price'].values.reshape(-1, 1)

            group_copy = group.copy()
            group_copy['Predicted_Price'] = scaler.inverse_transform(preds_2d).flatten()
            group_copy['Actual_Price'] = scaler.inverse_transform(actuals_2d).flatten()

            all_inverse_transformed.append(group_copy)

    # 3. Combine into one master results dataframe
    master_results_df = pd.concat(all_inverse_transformed, ignore_index=True)

    # 4. Save to a CSV file
    output_file = 'E:\\Portfolio Optimization & Market Trend Predictor repos\\reports\\all_companies_predictions.csv'
    master_results_df.to_csv(output_file, index=False)
    print(f'Master results successfully saved to {output_file}!')


if __name__ == "__main__":
    train_model()