import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.dates as mdates
from sklearn.metrics import ConfusionMatrixDisplay
import joblib
import os


def plot_price_accuracy(model_path, X_test, y_test):

    model = joblib.load(model_path)

    y_pred = model.predict(X_test)

    plt.figure(figsize=(8, 6))

    # Scatter plot of actual vs predicted
    plt.scatter(y_test, y_pred, alpha=0.7, color='teal', edgecolor='k')

    # Draw the perfect 1:1 fit line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot(
        [min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Fit'
    )

    plt.xlabel('Actual Price ($)', fontsize=11)
    plt.ylabel('Predicted Price ($)', fontsize=11)
    plt.title(
        'SpendSense Price Predictor: Actual vs. Predicted', fontsize=12, pad=15
    )
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    

    # Extract the filename
    filename = os.path.basename(model_path)

    name_only, extension = os.path.splitext(filename)


    save_path = ("E:\\Bank_Transaction_Categorization_Tool_repos\\reports\\plots\\" + name_only + "_scatterplot.png")

    plt.savefig(
    save_path, dpi=300, bbox_inches='tight'
    )  # dpi=300 makes it high-resolution
    print(f'Plot successfully saved to {save_path}')

    # Display the plot
    plt.show()


if __name__ == "__main__":
    print("Done")