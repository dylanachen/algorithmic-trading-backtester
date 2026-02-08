# algorithmic-trading-backtester
A systematic backtesting engine for algorithmic trading strategies, combining technical indicators, time-series models, and portfolio simulations to evaluate risk-adjusted returns.

## Setup

```bash
pip install -r requirements.txt
```

### Dependencies
    pandas
    yfinance
    requests
    statsmodels

## Download data
**yfinance (default, no API key)**
We have multiple stock data sources, but we will use yfinance at this time. If you want to see every sources that we prepared, you can check this [download data](https://github.com/dylanachen/algorithmic-trading-backtester/tree/main/data_collection)

```bash
cd ~/data_collection
python -m pip install -r requirements.txt
python download_stock_data.py --source yfinance --output-dir data
```




# Run Command
MA



ARIMA

