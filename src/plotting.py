import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.dates as mdates



# Ticker,Date,Predicted_Price,Actual_Price
def plot_company_preditions():
    current_dir = Path(__file__).resolve().parent
    root_dir = current_dir.parent
    data_path = root_dir / "reports"

    df = pd.read_csv("E:\\Portfolio Optimization & Market Trend Predictor repos\\reports\\all_companies_predictions.csv")  
    df['Date'] = pd.to_datetime(df['Date'])

    for ticker, group in df.groupby('Ticker'):
        plt.figure(figsize=(12, 6))
        plt.plot(group['Date'], group['Actual_Price'], label='Actual Price', color='blue')
        plt.plot(group['Date'], group['Predicted_Price'], label='Predicted Price', color='orange')
        
        # Save the plot to a file

        plt.savefig(data_path / f'{ticker}_predictions.png')
        plt.close()


if __name__ == "__main__":
    plot_company_preditions()   