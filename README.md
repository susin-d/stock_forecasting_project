# Stock Market Forecasting Dashboard

A beginner-friendly web application for stock price forecasting using multiple time series models. This project provides an intuitive interface for users to predict stock prices using ARIMA, SARIMA, Facebook Prophet, and LSTM models with interactive visualizations.

## ✨ Features

- **Beginner-Friendly UI**: Clean, modern interface with clear instructions and examples
- **Multiple Forecasting Models**:
  - ARIMA (AutoRegressive Integrated Moving Average)
  - SARIMA (Seasonal ARIMA)
  - Facebook Prophet
  - LSTM (Long Short-Term Memory) Neural Network
- **Interactive Dashboard**: Real-time stock data fetching and visualization
- **Model Evaluation**: RMSE and MAE metrics for model comparison
- **Responsive Design**: Works on desktop and mobile devices
- **Easy Setup**: Simple installation and deployment

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Internet connection (for fetching stock data)

### Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/susin-d/stock_forecasting_project.git
    cd stock_forecasting_project
    ```

2. **Create a virtual environment**:
    ```bash
    python -m venv venv
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the application**:
    ```bash
    python run.py
    ```

5. **Open your browser** and go to `http://localhost:8000`

## 📖 How to Use

### For Beginners:
1. **Enter a Stock Ticker**: Type a stock symbol like "AAPL" (Apple), "GOOGL" (Google), or "MSFT" (Microsoft)
2. **Choose a Model**: Select from ARIMA, SARIMA, Prophet, or LSTM
3. **Set Forecast Days**: Use the slider to choose how many days to forecast (1-120 days)
4. **Click "Run Forecast"**: Get instant predictions with visualizations

### Understanding the Results:
- **Forecast Chart**: Shows historical prices (green) and predicted prices (red)
- **RMSE/MAE Metrics**: Lower values indicate better model accuracy
- **Price Predictions**: Daily forecasted prices for your selected period

## 🏗️ Project Structure

```
stock_forecasting_project/
│
├── app/
│   ├── __init__.py               # Flask app initialization
│   ├── routes.py                 # Web routes and API endpoints
│   ├── templates/                # HTML templates
│   │   ├── index.html           # Main dashboard page
│   │   └── results.html         # Results display page
│   └── static/
│       └── styles.css           # Custom CSS styling
│
├── data/
│   ├── raw/                     # Raw stock data storage
│   └── README.md                # Data documentation
│
├── notebooks/
│   ├── 01_data_exploration.ipynb    # Data analysis notebook
│   ├── 02_preprocessing.ipynb       # Data preprocessing
│   ├── 03_arima_sarima.ipynb        # ARIMA/SARIMA models
│   ├── 04_prophet.ipynb            # Facebook Prophet
│   └── 05_lstm.ipynb              # LSTM neural network
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # Data loading utilities
│   ├── preprocessing.py         # Data preprocessing functions
│   ├── evaluation.py            # Model evaluation metrics
│   └── utils.py                 # Helper functions
│
├── requirements.txt             # Python dependencies
├── config.yaml                  # Configuration settings
├── run.py                       # Main application entry point
└── README.md                    # This file
```

## 📊 Models Overview

### ARIMA (AutoRegressive Integrated Moving Average)
- **Best for**: Stationary time series data
- **Use case**: Short-term predictions with clear patterns
- **Parameters**: (p, d, q) - autoregressive, differencing, moving average

### SARIMA (Seasonal ARIMA)
- **Best for**: Data with seasonal patterns
- **Use case**: Stocks with weekly/monthly cycles
- **Parameters**: Extends ARIMA with seasonal components

### Facebook Prophet
- **Best for**: Business time series with holidays and trends
- **Use case**: Long-term forecasts with external factors
- **Advantages**: Minimal preprocessing required

### LSTM (Long Short-Term Memory)
- **Best for**: Complex, non-linear patterns
- **Use case**: Capturing long-term dependencies in price movements
- **Advantages**: Deep learning approach for sophisticated predictions

## 🎯 Evaluation Metrics

- **RMSE (Root Mean Squared Error)**: Measures average prediction error magnitude
- **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual values
- **Lower is Better**: Both metrics indicate better model performance with smaller values

## 🔧 Configuration

Edit `config.yaml` to customize:
- Default stock tickers
- Model parameters
- Forecast horizons
- Data source settings

## 🐛 Troubleshooting

### Common Issues:

**Port 8000 already in use:**
```bash
# Kill process using port 8000
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

**Missing dependencies:**
```bash
pip install --upgrade -r requirements.txt
```

**Stock data not loading:**
- Check internet connection
- Verify ticker symbol is correct
- Some tickers may not be available

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Make your changes and test thoroughly
4. Commit your changes: `git commit -m 'Add new feature'`
5. Push to the branch: `git push origin feature/new-feature`
6. Submit a Pull Request

### Development Guidelines:
- Follow PEP 8 style guidelines
- Add docstrings to new functions
- Test your changes with different stock tickers
- Update documentation for new features

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Yahoo Finance** for providing free stock market data
- **Facebook** for the Prophet forecasting library
- **Statsmodels** for ARIMA/SARIMA implementations
- **PyTorch** for LSTM neural network support
- **Plotly** for interactive data visualizations

## 📞 Support

If you encounter any issues or have questions:

1. Check the troubleshooting section above
2. Review the existing issues on GitHub
3. Create a new issue with detailed information about your problem

---

**Happy Forecasting! 📈**
