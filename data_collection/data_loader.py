# Load stock CSV or fetch from yfinance / Alpha Vantage / Polygon. Expects Date, OHLCV.

import os
import pandas as pd
from typing import Dict, List, Optional, Any


def load_stock_csv(
    filepath: str,
    date_col: Optional[str] = None,
    date_format: Optional[str] = None,
) -> pd.DataFrame:
    """Load one stock CSV, detect date col."""
    df = pd.read_csv(filepath)
    df.columns = [c.strip() for c in df.columns]
    col_lower = {c.lower(): c for c in df.columns}
    if date_col:
        date_col_actual = date_col if date_col in df.columns else col_lower.get(date_col.lower())
    else:
        date_col_actual = col_lower.get("date") or col_lower.get("timestamp") or (df.columns[0] if "date" in df.columns[0].lower() else None)
    if date_col_actual:
        df[date_col_actual] = pd.to_datetime(df[date_col_actual], format=date_format)
        df = df.sort_values(date_col_actual).reset_index(drop=True)
        if "Date" not in df.columns and date_col_actual != "Date":
            df["Date"] = df[date_col_actual]
    def _find_col(needle: str) -> Optional[str]:
        needle_lower = needle.lower()
        if needle in df.columns:
            return needle
        for c in df.columns:
            if needle_lower in str(c).lower():
                return c
        return None
    for std_name, search in [("Close", "close"), ("Open", "open"), ("High", "high"), ("Low", "low"), ("Volume", "volume"), ("Date", "date")]:
        if std_name not in df.columns:
            found = col_lower.get(search) or _find_col(search)
            if found:
                if std_name == "Date":
                    df["Date"] = pd.to_datetime(df[found], errors="coerce")
                else:
                    df[std_name] = pd.to_numeric(df[found], errors="coerce")
    if "Date" in df.columns:
        df = df.sort_values("Date").reset_index(drop=True)
    return df


def load_portfolio_from_folder(
    folder: str,
    tickers: Optional[List[str]] = None,
    file_pattern: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """Load CSVs from folder. Ticker from filename (AAPL.csv or stock_AAPL.csv)."""
    if not os.path.isdir(folder):
        return {}
    result = {}
    for f in os.listdir(folder):
        if not f.endswith(".csv"):
            continue
        base = os.path.splitext(f)[0]
        ticker = base.split("_")[-1] if "_" in base else base
        if tickers is not None and ticker not in tickers:
            continue
        path = os.path.join(folder, f)
        try:
            result[ticker] = load_stock_csv(path)
        except Exception:
            continue
    return result


def _normalize_ohlcv_df(df: pd.DataFrame) -> pd.DataFrame:
    """Get Date, OHLCV columns."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    def _match_col(needle: str, col_name: str) -> bool:
        c = str(col_name).lower()
        return c == needle or needle in c
    if "Date" not in df.columns and "Datetime" in df.columns:
        df["Date"] = pd.to_datetime(df["Datetime"]).dt.strftime("%Y-%m-%d")
    elif "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    for name in ["Open", "High", "Low", "Close", "Volume"]:
        if name not in df.columns:
            for c in df.columns:
                if _match_col(name.lower(), c):
                    df[name] = pd.to_numeric(df[c], errors="coerce")
                    break
    if "Date" not in df.columns:
        for c in df.columns:
            if _match_col("date", c):
                df["Date"] = pd.to_datetime(df[c], errors="coerce")
                break
    keep = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    return df[keep] if keep else df


def fetch_yfinance(
    tickers: List[str],
    start: str = "2020-01-01",
    end: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """yfinance, no key."""
    try:
        import yfinance as yf
    except ImportError:
        print("Install yfinance: pip install yfinance")
        return {}
    if end is None:
        end = pd.Timestamp.today().strftime("%Y-%m-%d")
    data = {}
    for t in tickers:
        try:
            df = yf.download(t, start=start, end=end, progress=False, auto_adjust=True)
            if df.empty or len(df) < 2:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df = df.copy()
                df.columns = df.columns.get_level_values(0).astype(str)
            df = df.reset_index()
            df.columns = [str(c).replace(" ", "_") for c in df.columns]
            df = _normalize_ohlcv_df(df)
            if not df.empty:
                data[t] = df
        except Exception:
            continue
    return data


def fetch_alphavantage(
    tickers: List[str],
    start: str = "2020-01-01",
    end: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """Alpha Vantage daily. Need ALPHAVANTAGE_API_KEY."""
    api_key = api_key or os.environ.get("ALPHAVANTAGE_API_KEY")
    if not api_key:
        print("Alpha Vantage requires an API key. Set ALPHAVANTAGE_API_KEY or pass api_key=.")
        return {}
    try:
        import requests
    except ImportError:
        print("Install requests: pip install requests")
        return {}
    if end is None:
        end = pd.Timestamp.today().strftime("%Y-%m-%d")
    data = {}
    for t in tickers:
        try:
            url = (
                "https://www.alphavantage.co/query"
                "?function=TIME_SERIES_DAILY"
                f"&symbol={t}&outputsize=full&apikey={api_key}"
            )
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            j = r.json()
            ts = j.get("Time Series (Daily)") or j.get("time_series_daily")
            if not ts:
                continue
            rows = []
            for date_str, v in ts.items():
                if date_str < start or date_str > end:
                    continue
                rows.append({
                    "Date": date_str,
                    "Open": float(v.get("1. open", v.get("open", 0))),
                    "High": float(v.get("2. high", v.get("high", 0))),
                    "Low": float(v.get("3. low", v.get("low", 0))),
                    "Close": float(v.get("4. close", v.get("close", 0))),
                    "Volume": int(float(v.get("5. volume", v.get("volume", 0)))),
                })
            if not rows:
                continue
            df = pd.DataFrame(rows)
            df = df.sort_values("Date").reset_index(drop=True)
            data[t] = _normalize_ohlcv_df(df)
        except Exception as e:
            print(f"  Alpha Vantage {t}: {e}")
            continue
    return data


def fetch_polygon(
    tickers: List[str],
    start: str = "2020-01-01",
    end: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """Polygon daily. Need POLYGON_API_KEY."""
    api_key = api_key or os.environ.get("POLYGON_API_KEY")
    if not api_key:
        print("Polygon requires an API key. Set POLYGON_API_KEY or pass api_key=.")
        return {}
    try:
        import requests
    except ImportError:
        print("Install requests: pip install requests")
        return {}
    if end is None:
        end = pd.Timestamp.today().strftime("%Y-%m-%d")
    data = {}
    for t in tickers:
        try:
            url = (
                f"https://api.polygon.io/v2/aggs/ticker/{t}/range/1/day/{start}/{end}"
                f"?adjusted=true&sort=asc&limit=50000&apiKey={api_key}"
            )
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            j = r.json()
            results = j.get("results") or []
            if not results:
                continue
            rows = []
            for bar in results:
                ts_ms = bar.get("t", 0)
                date_str = pd.Timestamp(ts_ms, unit="ms").strftime("%Y-%m-%d")
                rows.append({
                    "Date": date_str,
                    "Open": float(bar.get("o", 0)),
                    "High": float(bar.get("h", 0)),
                    "Low": float(bar.get("l", 0)),
                    "Close": float(bar.get("c", 0)),
                    "Volume": int(bar.get("v", 0)),
                })
            df = pd.DataFrame(rows)
            data[t] = _normalize_ohlcv_df(df)
        except Exception as e:
            print(f"  Polygon {t}: {e}")
            continue
    return data


def fetch_from_api(
    tickers: List[str],
    start: str = "2020-01-01",
    end: Optional[str] = None,
    source: str = "yfinance",
    **kwargs: Any,
) -> Dict[str, pd.DataFrame]:
    """source: yfinance | alphavantage | polygon."""
    source = source.lower().strip()
    if source == "yfinance":
        return fetch_yfinance(tickers, start=start, end=end)
    if source == "alphavantage":
        return fetch_alphavantage(tickers, start=start, end=end, api_key=kwargs.get("api_key"))
    if source == "polygon":
        return fetch_polygon(tickers, start=start, end=end, api_key=kwargs.get("api_key"))
    print(f"Unknown source: {source}. Use yfinance, alphavantage, or polygon.")
    return {}


def fetch_sample_data(
    tickers: List[str],
    start: str = "2020-01-01",
    end: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """yfinance default."""
    return fetch_yfinance(tickers, start=start, end=end)


def normalize_date_str(x) -> str:
    """YYYY-MM-DD string so CSV and API dates match."""
    if x is None or pd.isna(x):
        return ""
    s = str(x).strip()
    if len(s) >= 10:
        return s[:10]
    return s


def get_common_dates(frames: Dict[str, pd.DataFrame], date_col: str = "Date") -> List[str]:
    """Dates that appear in every dataframe."""
    if not frames:
        return []
    sets = []
    for df in frames.values():
        if date_col in df.columns:
            sets.append(set(normalize_date_str(x) for x in df[date_col]))
        else:
            sets.append(set(normalize_date_str(x) for x in df.index))
    common = sets[0]
    for s in sets[1:]:
        common = common & s
    return sorted(d for d in common if d)
