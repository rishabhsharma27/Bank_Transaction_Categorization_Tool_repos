Data Source: https://www.kaggle.com/datasets/ahmedmohamed2003/spending-habits

Dataset Overview
The Spending Patterns Dataset provides a synthetic representation of consumer spending behavior across various categories. 
This dataset is ideal for exploratory data analysis, statistical modeling, and machine learning applications related to 
financial forecasting, customer segmentation, or consumer behavior analysis.

Price model overview:
For the price prediction model, I did some EDA and added another column with a range of unit prices. 
Then, using bins at 50$ intervel worked to reduce the error to upto 131.43 for items ranging from 0 - $10000, 
and with an interval of $100 to 113.58$, 
interval of $200 to $96.50, 
interval of $400 to $83.18, 
interval of 1000$ to $112.24, 
The RMSE is 'Min' at $83.18 for the interval.
