from statsmodels.tsa.arima.model import ARIMA
import pickle

def train_arima(series, order=(5,1,0)):
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    return model_fit

def save_model(model, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

def load_model(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)
