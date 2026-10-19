import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Load the saved master predictions file
df = pd.read_csv('E:\\Portfolio Optimization & Market Trend Predictor repos\\reports\\all_companies_predictions.csv')


# 2. Define a function to calculate metrics for a group of data
def evaluate_predictions(group):
  actual = group['Actual_Price']
  predicted = group['Predicted_Price']

  mae = mean_absolute_error(actual, predicted)
  rmse = np.sqrt(mean_squared_error(actual, predicted))
  # MAPE: Mean Absolute Percentage Error (%)
  mape = np.mean(np.abs((actual - predicted) / actual)) * 100

  return pd.Series({'MAE': mae, 'RMSE': rmse, 'MAPE(%)': mape})


# 3. Apply metrics calculation grouped by Ticker
metrics_df = df.groupby('Ticker').apply(evaluate_predictions).reset_index()

# Round for clean readability
metrics_df[['MAE', 'RMSE', 'MAPE(%)']] = metrics_df[['MAE', 'RMSE', 'MAPE(%)']].round(
    4
)

print('--- Model Performance Summary per Company ---')
print(metrics_df.to_string(index=False))

# 4. (Optional) Save evaluation metrics to a new CSV file
metrics_df.to_csv('E:\\Portfolio Optimization & Market Trend Predictor repos\\reports\\model_evaluation_metrics.csv', index=False)
print('\nEvaluation metrics successfully saved to model_evaluation_metrics.csv!')