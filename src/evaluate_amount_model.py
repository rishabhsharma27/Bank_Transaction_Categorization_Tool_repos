import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.dates as mdates
from sklearn.metrics import ConfusionMatrixDisplay
import joblib
import os
from sklearn.metrics import mean_squared_error



def eval_price_accuracy(model_path, X_test, y_test):

    model = joblib.load(model_path)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error (MSE): {mse:.2f}')

    # 3. Pro-Tip: Calculate Root Mean Squared Error (RMSE)
    # RMSE is usually preferred for price models because it is in the exact same units (dollars) as your target variable!
    rmse = np.sqrt(mse)
    print(f'Root Mean Squared Error (RMSE): ${rmse:.2f}')



if __name__ == "__main__":
    print("Done")