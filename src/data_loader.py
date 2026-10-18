import pandas as pd
import sys
from pathlib import Path


def sp500_year_split():
    # 1. Set up paths
    current_dir = Path(__file__).resolve().parent
    root_dir = current_dir.parent
    data_path = root_dir / "data"

    # Automatically finds your project root and adds it to Python's search path
    root_dir = Path(__file__).resolve().parent.parent
    sys.path.append(str(root_dir))

    # 1. Load your dataset
    df = pd.read_csv(data_path / 'sp500_stocks.csv')

    # 2. Ensure your date column is in datetime format
    date_col = 'date'
    df[date_col] = pd.to_datetime(df[date_col])

    # 3. Extract the year into a temporary column
    df['Year'] = df[date_col].dt.year

    # 4. Group by year and save each group to a separate file
    for year, group in df.groupby('Year'):
    # Optional: Drop the temporary 'Year' column if you want the original schema
        output_df = group.drop(columns=['Year'])
        
    # Define a clean filename for each year
        filename = data_path / f'sp500_stocks_{year}.csv'

    # Save to CSV (or .to_parquet() for compressed columnar storage)
        output_df.to_csv(filename, index=False)
        print(f'Saved: {filename} ({len(output_df):,} rows)')
    

# Test the function if running data_loader.py directly
if __name__ == "__main__":
    sp500_year_split()



