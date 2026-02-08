# DSCI-560: Data Science Practicum Laboratory Assignment 4
A systematic backtesting engine for algorithmic trading strategies, combining technical indicators, time-series models, and portfolio simulations to evaluate risk-adjusted returns.

## Project Overview

This project builds a multi-asset backtesting pipeline using daily OHLCV data downloaded with `yfinance` (one CSV per ticker). All tickers are aligned by the intersection of common trading dates to ensure decisions are made on the same days.

We evaluate on a 20-trading-day holdout period. These last 20 days are excluded from model fitting, so the training history never overlaps with the test window. On each holdout day `t`, we fit a separate forecasting model per ticker using data only up to `t-1` (no look-ahead) and predict future **Open** prices. We support MA and ARIMA baselines; with horizon 1, we predict only the next-day Open.

Tickers are scored by predicted return and the top `N` are selected at each rebalance. Trades execute at the **Open** price, with rebalancing every `K` days and a minimum holding constraint to reduce turnover. The system outputs transaction logs, performance metrics (including Sharpe ratio), and plots of portfolio value and rolling Sharpe.


## Team: pylovers

| Name | USC ID |
|------|--------|
| Dylan Chen | 6984540266 |
| Angela Kang | 8957777203 |
| Vincent-Daniel Yun | 4463771151 |


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

- `--data_dir`  
  Path to the folder that contains one CSV file per ticker (the trading universe).

- `--pred_method`  
  Which forecasting method to use to predict future Open prices (`naive`, `ma`, or `arima`).

- `--ma_window`  
  Lookback window length (in trading days) used by the moving-average predictor (only used when `--pred_method ma`).

- `--holdout_n`  
  Number of trading days in the holdout (backtest) period; the simulation runs only on the last `holdout_n` common dates.

- `--top_n`  
  How many tickers to select at each rebalance based on the forecast-based score.

- `--rebalance_every`  
  Rebalance frequency (in holdout trading days); selection is recomputed once every `rebalance_every` days.

- `--min_hold_days`  
  Minimum number of trading days a position must be held before it is allowed to be sold (except forced end liquidation).

- `--initial_cash`  
  Starting cash amount for the simulated portfolio.

- `--out_dir`  
  Output directory where result CSVs and plots are saved.




### MA Example

    python3 main.py \
      --data_dir ./data \
      --pred_method ma \
      --ma_window 60 \
      --holdout_n 20 \
      --top_n 10 \
      --rebalance_every 3 \
      --min_hold_days 2 \
      --initial_cash 1000000 \
      --out_dir out_lab4_topN



### ARIMA Example

    python3 main.py \
      --data_dir ./data \
      --pred_method arima \
      --arima_order 5,1,0 \
      --holdout_n 20 \
      --top_n 10 \
      --rebalance_every 3 \
      --min_hold_days 2 \
      --initial_cash 1000000 \
      --out_dir out_lab4_topN


# Final Results
### MA

    === Strategy ===
    Price alignment    : predict NEXT OPEN, trade at OPEN
    Method             : ma
    
    === Portfolio Metrics (holdout trading) ===
    Initial value     : $1,007,117.73
    Final value       : $888,299.76
    Total return      : -11.798%
    Annualized return : -79.439%
    Sharpe ratio      : -7.3250
    Max drawdown      : -12.175%
    Trading days      : 20
    


### ARIMA

    === Strategy ===
    Price alignment    : predict NEXT OPEN, trade at OPEN
    Method             : arima
    
    === Portfolio Metrics (holdout trading) ===
    Initial value     : $1,000,011.44
    Final value       : $991,151.83
    Total return      : -0.886%
    Annualized return : -10.607%
    Sharpe ratio      : -0.4516
    Max drawdown      : -5.408%
    Trading days      : 20



# Plots
### MA
<img width="2400" height="840" alt="Image" src="https://github.com/user-attachments/assets/abc4dda8-e07a-4649-a8a0-e82eb952bbdd" />


### ARIMA
<img width="2400" height="840" alt="Image" src="https://github.com/user-attachments/assets/e2527bd6-a933-4af2-9c0d-ceac1e480057" />
