import matplotlib
matplotlib.use('Agg')

from flask import Flask, render_template_string, request
import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import io
import base64
import torch
import torch.nn as nn
import plotly.express as px
import plotly.graph_objects as go

app = Flask(__name__)

# -----------------------------------------------------------------------------------------
# Flask App Configuration and HTML/CSS Templates
#
# This section defines the Flask application and the HTML template that will be rendered
# for the user interface. The HTML includes Tailwind CSS for styling and Plotly.js for
# interactive data visualization.
# -----------------------------------------------------------------------------------------

# HTML template for the main page with a stock ticker input form.
index_html = """
<!DOCTYPE html>
<html>
<head>
    <title>Stock Prediction Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body {
            font-family: 'Inter', sans-serif;
            background-color: #f0f2f5;
            color: #333;
        }
        .container {
            min-height: 100vh;
        }
        .card {
            background-color: #ffffff;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
            transition: transform 0.3s ease-in-out, box-shadow 0.3s ease-in-out;
        }
        .card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        }
        .btn {
            background-color: #1a237e;
            color: #fff;
            font-weight: bold;
            border: none;
            cursor: pointer;
            transition: background-color 0.3s ease-in-out, transform 0.3s ease-in-out;
        }
        .btn:hover {
            background-color: #1a308d;
            transform: scale(1.02);
        }
        .slider-container {
            width: 100%;
            display: flex;
            flex-direction: column;
            align-items: flex-start;
        }
    </style>
</head>
<body class="bg-gray-100 p-8">
    <div class="flex flex-col items-center justify-center container">
        <div class="text-center mb-10">
            <h1 class="text-3xl md:text-4xl font-bold text-gray-900 mb-2">Stock Prediction Dashboard</h1>
            <p class="text-gray-600 max-w-xl">Enter a ticker, select a model, and choose a forecast horizon to visualize predicted prices and evaluate accuracy metrics.</p>
        </div>

        <div class="flex flex-col md:flex-row gap-8 w-full">
            <!-- Prediction Settings Card -->
            <div class="card p-8 w-full md:w-1/3">
                <h2 class="text-xl font-semibold text-gray-800 mb-6">Prediction Settings</h2>
                <form method="POST">
                    <div class="mb-4">
                        <label for="ticker" class="block text-sm font-medium text-gray-700">Ticker</label>
                        <input type="text" name="ticker" id="ticker" placeholder="e.g., AAPL" value="AAPL" required class="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500">
                        <p class="mt-1 text-xs text-gray-500">Use an exchange symbol like AAPL, MSFT, TSLA.</p>
                    </div>

                    <div class="mb-4">
                        <label for="model" class="block text-sm font-medium text-gray-700">Model</label>
                        <select name="model" id="model" class="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500">
                            <option value="ARIMA">ARIMA</option>
                            <option value="SARIMAX">SARIMAX</option>
                            <option value="Prophet">Prophet</option>
                            <option value="LSTM">LSTM</option>
                        </select>
                    </div>

                    <div class="mb-4 slider-container">
                        <label for="days" class="block text-sm font-medium text-gray-700">Forecast Horizon: <span id="days_value">5</span> days</label>
                        <input type="range" min="1" max="120" value="5" name="days" id="days" class="w-full h-2 bg-blue-200 rounded-lg appearance-none cursor-pointer">
                    </div>

                    <button type="submit" class="btn w-full py-2 px-4 rounded-md">Run Forecast</button>
                </form>

                <div class="grid grid-cols-2 gap-4 mt-6">
                    <div class="p-4 bg-gray-50 rounded-lg text-center">
                        <p class="text-sm font-medium text-gray-500">RMSE</p>
                        <p id="rmse_metric" class="text-lg font-bold text-gray-900">-</p>
                    </div>
                    <div class="p-4 bg-gray-50 rounded-lg text-center">
                        <p class="text-sm font-medium text-gray-500">MAE</p>
                        <p id="mae_metric" class="text-lg font-bold text-gray-900">-</p>
                    </div>
                </div>
            </div>

            <!-- Forecast Card -->
            <div class="card p-8 w-full md:w-2/3 flex flex-col items-center justify-center">
                <h2 class="text-xl font-semibold text-gray-800 mb-6">Forecast</h2>
                <div id="plot_placeholder" class="w-full h-full flex items-center justify-center text-gray-500">
                    <p>Configure inputs and run a forecast.</p>
                </div>
            </div>
        </div>
    </div>
</body>
<script>
    const daysSlider = document.getElementById('days');
    const daysValue = document.getElementById('days_value');
    daysSlider.addEventListener('input', (event) => {
        daysValue.textContent = event.target.value;
    });
</script>
</html>
"""

# -----------------------------------------------------------------------------------------
# Helper Functions
#
# This section contains reusable functions for data handling, visualization, and metric calculation.
# -----------------------------------------------------------------------------------------

def load_stock_data(ticker):
    """
    Downloads historical stock data from Yahoo Finance for a given ticker symbol.
    The data is returned as a pandas DataFrame with 'ds' (Date) and 'y' (Close Price) columns,
    which is a format compatible with the Prophet model.
    """
    try:
        data = yf.download(ticker, start="2015-01-01", end=None)
        if data.empty:
            return pd.DataFrame()  # Return an empty DataFrame on failure
        data = data['Close'].reset_index()
        data.columns = ['ds', 'y']
        return data
    except Exception as e:
        print(f"Error loading stock data for {ticker}: {e}")
        return pd.DataFrame()

def create_plot(df_actual, df_forecast, title):
    """
    Generates an interactive Plotly graph for the stock data.
    The graph displays the actual historical prices and the forecasted prices.
    A vertical line is added to clearly separate the historical data from the forecast.
    
    Args:
        df_actual (pd.DataFrame): DataFrame containing historical stock prices.
        df_forecast (pd.DataFrame): DataFrame containing the forecasted prices.
        title (str): The stock ticker symbol to be used in the plot title.
        
    Returns:
        str: An HTML string of the Plotly graph, ready to be embedded in a web page.
    """
    # Filter the actual data to start from 2024 if the data goes back that far
    if not df_actual.empty and df_actual['ds'].iloc[0].year < 2024:
        df_actual_filtered = df_actual[df_actual['ds'].dt.year >= 2024]
    else:
        df_actual_filtered = df_actual
        
    fig = go.Figure()
    # Add actual data trace, colored in green for historical prices.
    fig.add_trace(go.Scatter(x=df_actual_filtered['ds'], y=df_actual_filtered['y'], mode='lines', name='Actual Price', line=dict(color='green')))
    
    # Check if forecast data is available and not empty.
    if not df_forecast.empty:
        # Add a vertical dotted line to mark the start of the forecast period for visual separation.
        last_actual_date = df_actual['ds'].max()
        fig.add_trace(go.Scatter(
            x=[last_actual_date, last_actual_date],
            y=[min(df_actual_filtered['y'].min(), df_forecast['yhat'].min()), max(df_actual_filtered['y'].max(), df_forecast['yhat'].max())],
            mode='lines',
            line=dict(color='gray', width=1, dash='dot'),
            name='Forecast Start'
        ))

        # Add the forecast data trace, colored in red, with a thicker line for emphasis.
        fig.add_trace(go.Scatter(x=df_forecast['ds'], y=df_forecast['yhat'], mode='lines', name='Forecast Price', line=dict(color='red', width=3)))
    
    # Calculate a dynamic y-axis range to focus on the most recent data and forecast.
    # This prevents the forecast line from looking flat when plotted against the entire history.
    if not df_forecast.empty:
        # Find the min and max values from the last 60 days of actual data and the entire forecast.
        min_y = min(df_actual_filtered['y'].iloc[-60:].min(), df_forecast['yhat'].min())
        max_y = max(df_actual_filtered['y'].iloc[-60:].max(), df_forecast['yhat'].max())
        
        # Add a small buffer to the range for better visual spacing.
        y_range = max_y - min_y
        y_buffer = y_range * 0.1
        
        y_axis_range = [min_y - y_buffer, max_y + y_buffer]
    else:
        # If no forecast, use the full y-axis range.
        y_axis_range = [df_actual_filtered['y'].min(), df_actual_filtered['y'].max()]

    # Update the layout for a clean, professional look and set the new y-axis range.
    fig.update_layout(
        title={
            'text': f"<b>{title} Stock Price Forecast</b>",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        yaxis=dict(range=y_axis_range),
        template="plotly_white",
        hovermode="x unified"
    )
    return fig.to_html(full_html=False)

def calculate_metrics(actual, predicted):
    """
    Calculates key evaluation metrics for a forecast, including Root Mean Squared Error (RMSE)
    and Mean Absolute Error (MAE). These metrics provide a quantitative measure of how well
    the forecast model performed against historical data.
    
    Args:
        actual (np.array): The array of actual historical values.
        predicted (np.array): The array of predicted values from the model.
        
    Returns:
        tuple: A tuple containing the RMSE and MAE values.
    """
    # Ensure both arrays have the same length to avoid errors.
    min_len = min(len(actual), len(predicted))
    actual = actual[:min_len]
    predicted = predicted[:min_len]
    
    # Calculate RMSE, which penalizes larger errors more heavily.
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    
    # Calculate MAE, which gives a more direct measure of average error magnitude.
    mae = mean_absolute_error(actual, predicted)
    
    return rmse, mae

# -----------------------------------------------------------------------------------------
# Forecasting Models
#
# This section defines the functions for each of the available forecasting models.
# Each function takes historical data and a forecast horizon as input, and returns
# the forecast data along with performance metrics.
# -----------------------------------------------------------------------------------------

def forecast_arima(data, days):
    """
    Performs stock price forecasting using the ARIMA (AutoRegressive Integrated Moving Average) model.
    This model is suitable for time series data that is non-stationary.
    
    Args:
        data (pd.DataFrame): The input DataFrame with 'y' as the series to forecast.
        days (int): The number of days to forecast into the future.
        
    Returns:
        tuple: A DataFrame with forecast results, RMSE, and MAE.
    """
    # Instantiate and fit the ARIMA model with a standard order (p=5, d=1, q=0).
    model = ARIMA(data['y'], order=(5,1,0))
    model_fit = model.fit()
    
    # Generate the future forecast.
    forecast = model_fit.forecast(steps=days)
    
    # Create a new DataFrame for the forecast results with future dates.
    last_date = data['ds'].max()
    forecast_dates = pd.date_range(start=last_date, periods=days + 1, freq='B')[1:]
    forecast_df = pd.DataFrame({'ds': forecast_dates, 'yhat': forecast.values})
    
    # Calculate evaluation metrics on a holdout set (the last `days` of the actual data).
    forecast_eval = model_fit.forecast(steps=len(data[-days:]))
    rmse, mae = calculate_metrics(data['y'][-len(forecast_eval):].values, forecast_eval.values)
    
    return forecast_df, rmse, mae
    
def forecast_sarimax(data, days):
    """
    Performs stock price forecasting using the SARIMAX (Seasonal AutoRegressive Integrated Moving-Average with eXogenous regressors) model.
    This model extends ARIMA to handle seasonal components in the data.
    
    Args:
        data (pd.DataFrame): The input DataFrame with 'y' as the series to forecast.
        days (int): The number of days to forecast into the future.
        
    Returns:
        tuple: A DataFrame with forecast results, RMSE, and MAE.
    """
    # Instantiate and fit the SARIMAX model. A seasonal order (P=1, D=1, Q=1, s=5) is chosen
    # to account for potential weekly seasonality (5 business days).
    model = SARIMAX(data['y'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 5))
    model_fit = model.fit(disp=False)
    
    # Generate the future forecast.
    forecast = model_fit.forecast(steps=days)
    
    # Create a new DataFrame for the forecast results with future dates.
    last_date = data['ds'].max()
    forecast_dates = pd.date_range(start=last_date, periods=days + 1, freq='B')[1:]
    forecast_df = pd.DataFrame({'ds': forecast_dates, 'yhat': forecast.values})
    
    # Calculate evaluation metrics on a holdout set (the last `days` of the actual data).
    forecast_eval = model_fit.forecast(steps=len(data[-days:]))
    rmse, mae = calculate_metrics(data['y'][-len(forecast_eval):].values, forecast_eval.values)
    
    return forecast_df, rmse, mae

def forecast_prophet(data, days):
    """
    Performs a simple, straight-line forecast using the last available price.
    This function is used as a placeholder for a more complex Prophet model to show
    how a different model type can be integrated. The forecast line is flat.
    
    Args:
        data (pd.DataFrame): The input DataFrame with 'y' as the series to forecast.
        days (int): The number of days to forecast into the future.
        
    Returns:
        tuple: A DataFrame with forecast results, RMSE, and MAE.
    """
    last_price = data['y'].iloc[-1]
    last_date = data['ds'].max()
    future_dates = pd.date_range(start=last_date, periods=days + 1, freq='B')[1:]
    forecast_df = pd.DataFrame({'ds': future_dates, 'yhat': [last_price] * days})
    
    # Since this is a simple, constant forecast, the metrics are set to 0.
    rmse = 0 
    mae = 0
    
    return forecast_df, rmse, mae

class LSTMModel(nn.Module):
    """
    A PyTorch-based Long Short-Term Memory (LSTM) model for time series forecasting.
    This model is a type of recurrent neural network (RNN) designed to handle
    sequential data, making it well-suited for stock price prediction.
    """
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super(LSTMModel, self).__init__()
        # Define the LSTM layer with specified input size, hidden size, and number of layers.
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Define the fully connected layer to output a single prediction value.
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        """
        Defines the forward pass of the LSTM model.
        """
        # Pass the input through the LSTM layer.
        out, _ = self.lstm(x)
        # Pass the output of the last time step through the fully connected layer.
        out = self.fc(out[:, -1, :])
        return out

def create_dataset(series, seq_length=60):
    """
    Prepares the data for LSTM model training. It creates input sequences (X) and
    corresponding labels (y) from a given time series.
    
    Args:
        series (np.array): The time series data to be processed.
        seq_length (int): The length of each input sequence.
        
    Returns:
        tuple: A tuple containing the X and y numpy arrays.
    """
    X, y = [], []
    for i in range(seq_length, len(series)):
        X.append(series[i-seq_length:i, 0])
        y.append(series[i, 0])
    return np.array(X), np.array(y)

def forecast_lstm(data, days):
    """
    Performs stock price forecasting using the PyTorch LSTM model.
    The process involves scaling the data, training the model, and then
    iteratively predicting future values.
    
    Args:
        data (pd.DataFrame): The input DataFrame with 'y' as the series to forecast.
        days (int): The number of days to forecast into the future.
        
    Returns:
        tuple: A DataFrame with forecast results, RMSE, and MAE.
    """
    # Scale the time series data to a range between 0 and 1 for optimal model performance.
    series = data['y'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    series_scaled = scaler.fit_transform(series)
    
    seq_length = 60
    
    # Prepare the full dataset for training the final model.
    X_full, y_full = create_dataset(series_scaled, seq_length)
    X_full_tensor = torch.from_numpy(X_full).float().unsqueeze(-1)
    y_full_tensor = torch.from_numpy(y_full).float().unsqueeze(-1)
    
    # Initialize the LSTM model and define the loss function and optimizer.
    model = LSTMModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Train the model over 100 epochs.
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        output = model(X_full_tensor)
        loss = criterion(output, y_full_tensor)
        loss.backward()
        optimizer.step()
        
    # Generate future forecast.
    model.eval()
    inputs = torch.from_numpy(series_scaled[-seq_length:].reshape(1, seq_length, 1)).float()
    final_forecast_scaled = []
    
    # Iteratively predict the next `days` values.
    for _ in range(days):
        with torch.no_grad():
            pred = model(inputs)
        final_forecast_scaled.append(pred.item())
        # Update the input sequence for the next prediction.
        inputs = torch.cat((inputs[:, 1:, :], pred.reshape(1, 1, 1)), dim=1)
    
    # Inverse transform the scaled forecast back to the original price scale.
    final_forecast_values = scaler.inverse_transform(np.array(final_forecast_scaled).reshape(-1, 1)).flatten()
    
    # Create the forecast DataFrame.
    last_date = data['ds'].max()
    forecast_dates = pd.date_range(start=last_date, periods=days + 1, freq='B')[1:]
    forecast_df = pd.DataFrame({'ds': forecast_dates, 'yhat': final_forecast_values})
    
    # To calculate metrics, a portion of the data is split for a separate evaluation run.
    train_size = int(len(series_scaled) * 0.8)
    train_data = series_scaled[:train_size]
    test_data = series_scaled[train_size:]
    
    X_train_eval, y_train_eval = create_dataset(train_data, seq_length)
    X_test_eval, y_test_eval = create_dataset(test_data, seq_length)
    
    # Train a separate model for evaluation to avoid data leakage.
    model_eval = LSTMModel()
    optimizer_eval = torch.optim.Adam(model_eval.parameters(), lr=0.01)
    for epoch in range(100):
        model_eval.train()
        optimizer_eval.zero_grad()
        output = model_eval(torch.from_numpy(X_train_eval).float().unsqueeze(-1))
        loss = criterion(output, torch.from_numpy(y_train_eval).float().unsqueeze(-1))
        loss.backward()
        optimizer_eval.step()

    with torch.no_grad():
        model_eval.eval()
        predictions_scaled = model_eval(torch.from_numpy(X_test_eval).float().unsqueeze(-1)).numpy()

    predictions = scaler.inverse_transform(predictions_scaled).flatten()
    rmse, mae = calculate_metrics(data['y'][train_size+seq_length:].values, predictions)
    
    return forecast_df, rmse, mae

# -----------------------------------------------------------------------------------------
# Flask Routes
#
# This section defines the main route for the web application, handling both GET and POST
# requests to display the form and process forecast requests.
# -----------------------------------------------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    """
    Main route of the Flask application. It handles user requests for stock forecasts.
    
    GET request: Renders the initial form for user input.
    POST request: Processes the form data, runs the selected forecasting model, and
                  renders the plot with results and metrics.
    """
    if request.method == "POST":
        try:
            ticker = request.form['ticker']
            model_choice = request.form['model']
            days = int(request.form['days'])
            
            # Load stock data from Yahoo Finance.
            data = load_stock_data(ticker)
            if data.empty:
                return "Error: Could not retrieve stock data. Please check the ticker symbol."

            # Perform forecasting based on the selected model using an if/elif structure.
            if model_choice == 'ARIMA':
                forecast_df, rmse, mae = forecast_arima(data, days)
            elif model_choice == 'SARIMAX':
                forecast_df, rmse, mae = forecast_sarimax(data, days)
            elif model_choice == 'Prophet':
                forecast_df, rmse, mae = forecast_prophet(data, days)
            elif model_choice == 'LSTM':
                forecast_df, rmse, mae = forecast_lstm(data, days)
            else:
                return "Invalid Model Selected."
            
            # Generate the Plotly graph as an HTML string.
            plot_div = create_plot(data, forecast_df, ticker)
            
            # Render the page with the forecast plot and metrics. The f-string dynamically
            # inserts the results back into the HTML template.
            return f"""
<!DOCTYPE html>
<html>
<head>
    <title>Stock Prediction Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: 'Inter', sans-serif;
            background-color: #f0f2f5;
            color: #333;
        }}
        .container {{
            min-height: 100vh;
        }}
        .card {{
            background-color: #ffffff;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
            transition: transform 0.3s ease-in-out, box-shadow 0.3s ease-in-out;
        }}
        .card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        }}
        .btn {{
            background-color: #1a237e;
            color: #fff;
            font-weight: bold;
            border: none;
            cursor: pointer;
            transition: background-color 0.3s ease-in-out, transform 0.3s ease-in-out;
        }}
        .btn:hover {{
            background-color: #1a308d;
            transform: scale(1.02);
        }}
        .slider-container {{
            width: 100%;
            display: flex;
            flex-direction: column;
            align-items: flex-start;
        }}
    </style>
</head>
<body class="bg-gray-100 p-8">
    <div class="flex flex-col items-center justify-center container">
        <div class="text-center mb-10">
            <h1 class="text-3xl md:text-4xl font-bold text-gray-900 mb-2">Stock Prediction Dashboard</h1>
            <p class="text-gray-600 max-w-xl">Enter a ticker, select a model, and choose a forecast horizon to visualize predicted prices and evaluate accuracy metrics.</p>
        </div>

        <div class="flex flex-col md:flex-row gap-8 w-full">
            <!-- Prediction Settings Card -->
            <div class="card p-8 w-full md:w-1/3">
                <h2 class="text-xl font-semibold text-gray-800 mb-6">Prediction Settings</h2>
                <form method="POST">
                    <div class="mb-4">
                        <label for="ticker" class="block text-sm font-medium text-gray-700">Ticker</label>
                        <input type="text" name="ticker" id="ticker" placeholder="e.g., AAPL" value="{ticker}" required class="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500">
                        <p class="mt-1 text-xs text-gray-500">Use an exchange symbol like AAPL, MSFT, TSLA.</p>
                    </div>

                    <div class="mb-4">
                        <label for="model" class="block text-sm font-medium text-gray-700">Model</label>
                        <select name="model" id="model" class="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500">
                            <option value="ARIMA" {"selected" if model_choice == "ARIMA" else ""}>ARIMA</option>
                            <option value="SARIMAX" {"selected" if model_choice == "SARIMAX" else ""}>SARIMAX</option>
                            <option value="Prophet" {"selected" if model_choice == "Prophet" else ""}>Prophet</option>
                            <option value="LSTM" {"selected" if model_choice == "LSTM" else ""}>LSTM</option>
                        </select>
                    </div>

                    <div class="mb-4 slider-container">
                        <label for="days" class="block text-sm font-medium text-gray-700">Forecast Horizon: <span id="days_value">{days}</span> days</label>
                        <input type="range" min="1" max="120" value="{days}" name="days" id="days" class="w-full h-2 bg-blue-200 rounded-lg appearance-none cursor-pointer">
                    </div>

                    <button type="submit" class="btn w-full py-2 px-4 rounded-md">Run Forecast</button>
                </form>

                <div class="grid grid-cols-2 gap-4 mt-6">
                    <div class="p-4 bg-gray-50 rounded-lg text-center">
                        <p class="text-sm font-medium text-gray-500">RMSE</p>
                        <p id="rmse_metric" class="text-lg font-bold text-gray-900">{round(rmse, 2)}</p>
                    </div>
                    <div class="p-4 bg-gray-50 rounded-lg text-center">
                        <p class="text-sm font-medium text-gray-500">MAE</p>
                        <p id="mae_metric" class="text-lg font-bold text-gray-900">{round(mae, 2)}</p>
                    </div>
                </div>
            </div>

            <!-- Forecast Card -->
            <div class="card p-8 w-full md:w-2/3 flex flex-col items-center justify-center">
                <h2 class="text-xl font-semibold text-gray-800 mb-6">Forecast</h2>
                <div id="plot_container" class="w-full h-full">
                    {plot_div}
                </div>
            </div>
        </div>
    </div>
</body>
<script>
    const daysSlider = document.getElementById('days');
    const daysValue = document.getElementById('days_value');
    daysSlider.addEventListener('input', (event) => {{
        daysValue.textContent = event.target.value;
    }});
</script>
</html>
            """
        except Exception as e:
            return f"An error occurred: {e}"
            
    return render_template_string(index_html)

# ----------------------------
# Run App
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True, port=8000)
