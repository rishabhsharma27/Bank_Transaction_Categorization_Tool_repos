import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from pathlib import Path
import joblib
import sys

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from src.plotting_price_model import plot_price_accuracy



def train_amount():
    current_dir = Path(__file__).resolve().parent
    root_dir = current_dir.parent
    data_path = root_dir / "data"
    model_path = root_dir / "models"

    df = pd.read_csv(data_path / "spending_patterns_detailed.csv")  # Ensure your CSV has 'Date' and 'Close' columns   

    bins = [0, 100, 1000, 10000, float('inf')]
    labels = ['100', '1000', '10000', '100000']

    # Apply custom cut
    df['Range'] = pd.cut(df['Total Spent'], bins=bins, labels=labels, right=True)

    #df['Payment category'] = df['Payment Method'].replace({'Credit Card': 'Card', 'Debit Card': 'Card'})

    df['Item_join'] = df[['Item', 'Category', 'Location']].agg(' '.join, axis=1)

    df = df[df['Total Spent']< 1000]

    X = df[['Item_join','Item']]
    y = df['Price Per Unit']

    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
    )

    # 2. Build Pipeline: One-Hot Encode categories/items -> Ridge Regression
    model = Pipeline([
        (
            'encoder',
            OneHotEncoder(handle_unknown='ignore'),
        ),  # Handles unseen categories smoothly
        ('regressor', Ridge(alpha=1.0)),  # Ridge handles sparse, high-dim data
    ])

    # 3. Fit the model
    model.fit(X, y)

    # 4. Predict the price of an item
    #new_data = pd.DataFrame([{'Category': 'Electronics', 'Item': 'Mechanical Keyboard'}])
    #predicted_price = model.predict(new_data)

    #print(f"Category: 'Electronics', Item: 'Mechanical Keyboard' => Predicted Price: ${predicted_price[0]:.2f}")

    # 'model' is your trained scikit-learn Pipeline
    # Save it to a file named 'price_model.pkl' (or 'spendsense_model.pkl')
    joblib.dump(model, (model_path / 'Amount_model.pkl'))
    
    plot_price_accuracy((model_path / 'Amount_model.pkl'), X_test, y_test)



if __name__ == "__main__":
    train_amount()