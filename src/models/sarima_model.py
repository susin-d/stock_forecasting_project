from statsmodels.tsa.statespace.sarimax import SARIMAX
import pickle

def train_sarima(series, order=(1,1,1), seasonal_order=(1,1,1,12)):
    model = SARIMAX(series, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit()
    return model_fit

def save_model(model, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

def load_model(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)
