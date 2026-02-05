# Lab 4 – Data Collection

Stock price data: load from CSV or download from APIs (yfinance, Alpha Vantage, Polygon).

## What's here

- **data_loader.py** – Load CSVs from a folder (`load_portfolio_from_folder`, `load_stock_csv`) or fetch OHLCV from APIs (`fetch_from_api`, `fetch_yfinance`, `fetch_alphavantage`, `fetch_polygon`). Output is always DataFrames with Date, Open, High, Low, Close, Volume.
- **download_stock_data.py** – Script to download data and save one CSV per ticker in `data/`.

## Setup

```bash
pip install -r requirements.txt
```

## Download data

```bash
python download_stock_data.py
```

Defaults: ~85 tickers across sectors (tech, finance, consumer, healthcare, industrials, energy, ETFs); start 2020-01-01; source **yfinance** (no API key). Use `--tickers` to override with a smaller set.

### Data sources

| Source         | Default? | API key? | Notes |
|----------------|----------|----------|-------|
| **yfinance**   | Yes      | No       | Full history where available. |
| Alpha Vantage  | No       | Yes (`ALPHAVANTAGE_API_KEY`) | **Free tier uses `outputsize=compact` (~100 recent trading days only) and has strict rate limits.** |
| Polygon        | No       | Yes (`POLYGON_API_KEY`)      | Full history within your Polygon plan limits. |

By default, downloads use **yfinance**. To use another source, use the copy-paste commands below.

### Copy-paste commands (all 3 options)

Run from the `data_collection` folder (e.g. `cd ~/data_collection` or your project path). Install once: `pip install -r requirements.txt` or `python -m pip install -r requirements.txt`.

**1. yfinance (default, no API key)**

```bash
cd ~/data_collection
python -m pip install -r requirements.txt
python download_stock_data.py --source yfinance --output-dir data
```

Smaller set (e.g. 3 tickers):

```bash
cd ~/data_collection
python download_stock_data.py --source yfinance --tickers AAPL MSFT GOOGL --output-dir data
```

**2. Alpha Vantage (requires API key, recent window only on free tier)**

Set your key first (PowerShell): `$env:ALPHAVANTAGE_API_KEY = "YOUR_KEY"`  
Or get a free key at: https://www.alphavantage.co/support/#api-key  

**Important limitations for this lab setup:**
- The code uses `outputsize=compact` so it works on the free tier → you only get the last ~100 trading days of daily data.  
- Alpha Vantage free keys have strict rate limits (roughly 1 request/second and low daily caps). If you hit the limit, you'll see “Thank you for using Alpha Vantage…” messages and no CSVs will be written until the limit resets.

```bash
cd ~/data_collection
python -m pip install -r requirements.txt
python download_stock_data.py --source alphavantage --output-dir data
```

With a few tickers:

```bash
cd ~/data_collection
python download_stock_data.py --source alphavantage --tickers AAPL MSFT GOOGL --output-dir data
```

**3. Polygon (requires API key)**

Set your key first (PowerShell): `$env:POLYGON_API_KEY = "YOUR_KEY"`  
Or sign up at: https://polygon.io/

```bash
cd ~/data_collection
python -m pip install -r requirements.txt
python download_stock_data.py --source polygon --output-dir data
```

With a few tickers:

```bash
cd ~/data_collection
python download_stock_data.py --source polygon --tickers AAPL MSFT GOOGL --output-dir data
```

### Options

| Option         | Example                          | Description |
|----------------|----------------------------------|-------------|
| `--tickers`    | `--tickers AAPL MSFT GOOGL`      | Ticker symbols (default: ~85 tickers) |
| `--start`     | `--start 2020-01-01`             | Start date (YYYY-MM-DD) |
| `--end`       | `--end 2024-12-31`               | End date (default: today) |
| `--output-dir`| `--output-dir data`              | Folder for CSV files |
| `--source`     | `--source yfinance`              | `yfinance`, `alphavantage`, or `polygon` |
| `--api-key`    | `--api-key YOUR_KEY`             | API key (or set env var for chosen source) |

## Use the loader in code

```python
from data_loader import load_portfolio_from_folder, fetch_from_api

# From CSV folder
price_data = load_portfolio_from_folder("data", tickers=["AAPL", "MSFT"])

# Or fetch from API
price_data = fetch_from_api(["AAPL", "MSFT"], start="2020-01-01", source="yfinance")
```
