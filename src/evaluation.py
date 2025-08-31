import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def plot_forecast(y_true, y_pred, title='Forecast vs Actual'):
    plt.figure(figsize=(12,6))
    plt.plot(y_true, label='Actual')
    plt.plot(y_pred, label='Forecast')
    plt.title(title)
    plt.legend()
    plt.show()
