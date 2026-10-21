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



def train_category_model():
    current_dir = Path(__file__).resolve().parent
    root_dir = current_dir.parent
    data_path = root_dir / "data"
    model_path = root_dir / "models"
    df = pd.read_csv(data_path / "spending_patterns_detailed.csv")  # Ensure your CSV has 'Date' and 'Close' columns

    # 2. Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        df['Item'], df['Category'], test_size=0.2, random_state=42
    )

    # 3. Build a Machine Learning Pipeline
    # A pipeline bundles text vectorization (TF-IDF) and the classifier together cleanly
    model_pipeline = Pipeline([
        (
            'tfidf',
            TfidfVectorizer(
                ngram_range=(1, 2), lowercase=True
            ),  # Captures single words and 2-word phrases
        ),
        ('classifier', LogisticRegression()),
    ])

    # 4. Train the model on raw text descriptions
    model_pipeline.fit(X_train, y_train)

    # 5. Test the model on unseen descriptions
    predictions = model_pipeline.predict(X_test)
    print('--- Test Predictions ---')
    for desc, pred in zip(X_test, predictions):
        print(f'Text: "{desc}" ---> Predicted Category: {pred}')

    # 'model' is your trained scikit-learn Pipeline
    # Save it to a file named 'price_model.pkl' (or 'spendsense_model.pkl')
    joblib.dump(model_pipeline, (model_path / 'Category_model.pkl'))
    
    print('Model saved successfully!')

    plot_model_accuracy((model_path / 'Category_model.pkl'), X_test, y_test)




if __name__ == "__main__":
    train_category_model()