from flask import render_template, request, jsonify
from app import app

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Load model, get data, run prediction
    return jsonify({'message': 'Prediction placeholder'})
