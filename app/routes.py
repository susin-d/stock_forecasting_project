from flask import render_template, request, jsonify
from app import app

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    ticker = request.form.get('ticker')
    # Load model, get data, run prediction
    # For now, placeholder data
    forecast_data = {
        'ticker': ticker.upper(),
        'current_price': 150.00,  # placeholder
        'forecast': [152.00, 155.00, 158.00]  # placeholder forecast
    }
    return render_template('results.html', data=forecast_data)
