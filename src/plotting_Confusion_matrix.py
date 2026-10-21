import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.dates as mdates
from sklearn.metrics import ConfusionMatrixDisplay
import joblib
import os
from sklearn.model_selection import train_test_split

def plot_model_accuracy(model_path, X_test, y_test):

    model = joblib.load(model_path)
    fig, ax = plt.subplots(figsize=(8, 6))

    ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        ax=ax,
        cmap='Blues',  # Color theme
        xticks_rotation=45,
        colorbar=True,
    )

    plt.title('SpendSense: Model Categorization Performance', fontsize=12, pad=15)
    plt.xlabel('Predicted Category', fontsize=10)
    plt.ylabel('Actual Category', fontsize=10)
    plt.tight_layout()

    # Extract the filename
    filename = os.path.basename(model_path)

    name_only, extension = os.path.splitext(filename)


    save_path = ("E:\\Bank_Transaction_Categorization_Tool_repos\\reports\\plots\\" + name_only + "_confusion_matrix.png")

    plt.savefig(
    save_path, dpi=300, bbox_inches='tight'
    )  # dpi=300 makes it high-resolution
    print(f'Plot successfully saved to {save_path}')

    # Display the plot
    plt.show()


if __name__ == "__main__":
    print("Done")