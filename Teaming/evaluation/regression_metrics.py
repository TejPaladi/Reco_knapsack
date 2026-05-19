import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Load data from CSV files, with an option to limit the number of rows
def load_data(file_path, nrows=None):
    return pd.read_csv(file_path, nrows=nrows)

# Function to calculate regression metrics directly
def calculate_metrics_directly(actual_scores, predicted_scores):
    # Calculate MAE, MSE, RMSE
    mae = mean_absolute_error(actual_scores, predicted_scores)
    mse = mean_squared_error(actual_scores, predicted_scores)
    rmse = np.sqrt(mse)
    
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse}

# File paths to your datasets
homedir = '../data/v0_output_teaming/teaming_434proposals_200researchers/'
data_files = {
    'M0': homedir + 'data_uc1_m2/m2_goodness_scores.csv',
    'M3': homedir + 'data_uc1_m3/m3_goodness_scores.csv'
}

# Number of rows to load
n=854136

# Load data
data_m0 = load_data(data_files['M0'], nrows=n)
data_m3 = load_data(data_files['M3'], nrows=n)

# Assuming 'goodness' is the column name for the scores in both files
# Ensure both dataframes have the same length
if len(data_m0) == len(data_m3):
    results_m0_m3 = calculate_metrics_directly(data_m3['goodness'], data_m0['goodness'])
    print("Metrics for M0 vs M3:", results_m0_m3)
else:
    print("Error: The number of rows in M0 and M3 does not match.")
