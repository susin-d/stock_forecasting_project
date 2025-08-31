import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import pickle

def create_lstm(window=30):
    model = Sequential([
        LSTM(50, input_shape=(window, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def save_model(model, filepath):
    model.save(filepath)

def load_model(filepath):
    from tensorflow.keras.models import load_model
    return load_model(filepath)
