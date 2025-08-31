import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def fill_missing(df):
    return df.fillna(method='ffill')

def scale_column(df, column):
    scaler = MinMaxScaler()
    df[column + '_scaled'] = scaler.fit_transform(df[[column]])
    return df, scaler

def difference(df, column):
    df[column + '_diff'] = df[column].diff()
    return df
