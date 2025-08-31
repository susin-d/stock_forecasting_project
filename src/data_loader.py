import pandas as pd

def load_stock_data(filepath):
    return pd.read_csv(filepath, parse_dates=['Date'])

def save_stock_data(df, filepath):
    df.to_csv(filepath, index=False)
