from prophet import Prophet
import pickle

def train_prophet(df):
    model = Prophet()
    model.fit(df)
    return model

def save_model(model, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

def load_model(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)
