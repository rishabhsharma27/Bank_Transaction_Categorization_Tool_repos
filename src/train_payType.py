import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from pathlib import Path
import joblib
import sys

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from src.plotting_Confusion_matrix import plot_model_accuracy


def train_payType():
    current_dir = Path(__file__).resolve().parent
    root_dir = current_dir.parent
    data_path = root_dir / "data"
    model_path = root_dir / "models"

    df = pd.read_csv(data_path / "spending_patterns_detailed.csv")  # Ensure your CSV has 'Date' and 'Close' columns   

    bins = [0, 100, 1000, 10000, float('inf')]
    labels = ['100', '1000', '10000', '100000']

    # Apply custom cut
    df['Range'] = pd.cut(df['Total Spent'], bins=bins, labels=labels, right=True)

    #df['Payment category'] = df['Payment Method'].replace({'Credit Card': 'Card', 'Debit Card': 'Card', 'Digital Wallet': 'Card'})

    df['Item_join'] = df[['Item', 'Location']].agg(' '.join, axis=1)

    df = df[(df['Total Spent']< 3000) & (df['Payment Method'] != 'Cash')]
    
    X_train, X_test, y_train, y_test = train_test_split(
        (df['Item_join']), df['Payment Method'], test_size=0.3, random_state=42
    )

    # 2. Pipeline setup

    model_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
        ('classifier', LogisticRegression()),
    ])

    # 3. Train the model to classify Online vs In-Store
    model_pipeline.fit(X_train, y_train)

    # 4. Test a new unseen transaction string
    predictions = model_pipeline.predict(X_test)

    print(f'"{X_test}" => Predicted Type: {predictions}')

    # 'model' is your trained scikit-learn Pipeline
    # Save it to a file named 'price_model.pkl' (or 'spendsense_model.pkl')
    joblib.dump(model_pipeline, (model_path / 'PayType_model.pkl'))
    
    print('Model saved successfully!')

    plot_model_accuracy((model_path / 'PayType_model.pkl'), X_test, y_test)



if __name__ == "__main__":
    train_payType()