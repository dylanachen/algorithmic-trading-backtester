# Download stock data to data/ CSVs. --source yfinance|alphavantage|polygon

import os
import argparse

from data_loader import fetch_from_api

# Default: large set across sectors (~80 tickers + ETFs)
DEFAULT_TICKERS = [
    # Tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "INTC", "CRM",
    "NFLX", "ADBE", "ORCL", "CSCO", "AVGO", "QCOM", "TXN", "IBM", "NOW", "INTU",
    "AMAT", "MU", "LRCX", "KLAC", "SNPS", "CDNS", "PANW", "CRWD", "SNOW", "MDB",
    # Finance
    "JPM", "BAC", "WFC", "C", "GS", "MS", "V", "MA", "AXP", "BLK", "SCHW", "BK",
    # Consumer
    "WMT", "HD", "MCD", "NKE", "SBUX", "COST", "TGT", "LOW", "TJX", "DG", "ROST",
    "PG", "KO", "PEP", "PM", "MO", "CL", "EL", "DIS", "CMCSA",
    # Healthcare
    "JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY", "TMO", "ABT", "DHR", "BMY", "AMGN",
    "GILD", "VRTX", "REGN", "MRNA", "ISRG", "SYK", "MDT", "ZTS",
    # Industrials / Energy / Other
    "CAT", "DE", "HON", "UPS", "BA", "LMT", "RTX", "GE", "MMM", "UNP",
    "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX",
    # ETFs
    "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO",
]


def download_and_save(
    tickers=None,
    start="2020-01-01",
    end=None,
    output_dir="data",
    source="yfinance",
    api_key=None,
):
    """Fetch OHLCV, save as CSV per ticker."""
    if tickers is None:
        tickers = DEFAULT_TICKERS
    base = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base, output_dir)
    os.makedirs(data_dir, exist_ok=True)

    data = fetch_from_api(
        tickers=tickers,
        start=start,
        end=end,
        source=source,
        api_key=api_key,
    )
    if not data:
        return False
    for ticker, df in data.items():
        if df is None or df.empty:
            continue
        path = os.path.join(data_dir, f"{ticker}.csv")
        df.to_csv(path, index=False)
        print(f"  Saved {path} ({len(df)} rows)")
    return len(data) > 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download stock data to data/")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=DEFAULT_TICKERS,
        help="Ticker symbols (default: ~85 tickers across sectors + ETFs)",
    )
    parser.add_argument(
        "--source",
        choices=["yfinance", "alphavantage", "polygon"],
        default="yfinance",
        help=(
            "Data source: yfinance (no key, full history where available), "
            "alphavantage (ALPHAVANTAGE_API_KEY, free tier: recent ~100 trading days only, "
            "strict rate limits), polygon (POLYGON_API_KEY)"
        ),
    )
    parser.add_argument("--start", default="2020-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (default: today)")
    parser.add_argument("--output-dir", default="data", help="Output folder for CSV files")
    parser.add_argument("--api-key", default=None, help="API key (or set env var for the chosen source)")
    args = parser.parse_args()

    api_key = args.api_key
    if not api_key and args.source == "alphavantage":
        api_key = os.environ.get("ALPHAVANTAGE_API_KEY")
    if not api_key and args.source == "polygon":
        api_key = os.environ.get("POLYGON_API_KEY")

    print(f"Downloading from {args.source}...")
    print(f"  Tickers: {args.tickers}  Start: {args.start}")
    success = download_and_save(
        tickers=args.tickers,
        start=args.start,
        end=args.end,
        output_dir=args.output_dir,
        source=args.source,
        api_key=api_key,
    )
    if success:
        print("Done. CSV files are in " + os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output_dir))
