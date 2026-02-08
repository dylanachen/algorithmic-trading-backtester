import os
import sys
import argparse
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from trading.portfolio import Portfolio
from trading.metrics import compute_metrics


def list_csv_files(data_dir: str) -> List[str]:
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    files = [f for f in os.listdir(data_dir) if f.lower().endswith(".csv")]
    files.sort()
    return files


def load_price_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {"Date", "Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(df.columns):
        missing = sorted(list(required - set(df.columns)))
        raise ValueError(f"CSV missing columns: {missing}. Found: {list(df.columns)}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).copy()
    df = df.sort_values("Date").reset_index(drop=True)

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Open", "Close"]).copy().reset_index(drop=True)
    if len(df) < 10:
        raise ValueError("Not enough rows after cleaning.")

    return df


def build_universe(data_dir: str) -> Dict[str, pd.DataFrame]:
    files = list_csv_files(data_dir)
    if not files:
        raise FileNotFoundError(f"No .csv files in {data_dir}")

    universe: Dict[str, pd.DataFrame] = {}
    bad = 0
    for f in files:
        path = os.path.join(data_dir, f)
        ticker = os.path.splitext(os.path.basename(path))[0]
        try:
            df = load_price_csv(path)
            universe[ticker] = df
        except Exception as e:
            bad += 1
            print(f"[skip] {f}: {e}")

    if len(universe) == 0:
        raise RuntimeError("No valid CSV files loaded.")
    if bad > 0:
        print(f"[info] skipped {bad} invalid files")

    return universe


def common_dates(universe: Dict[str, pd.DataFrame]) -> List[pd.Timestamp]:
    sets = []
    for _, df in universe.items():
        sets.append(set(pd.to_datetime(df["Date"]).tolist()))
    inter = set.intersection(*sets) if sets else set()
    return sorted(list(inter))


def df_by_date(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"])
    out = out.sort_values("Date").reset_index(drop=True)
    out = out.set_index("Date", drop=True)
    return out


def predict_multi_step_opens(
    history_open: pd.Series,
    method: str,
    ma_window: int,
    arima_order: Tuple[int, int, int],
    horizon: int,
) -> List[float]:
    h = int(max(1, horizon))
    hist = pd.to_numeric(history_open, errors="coerce").dropna().astype(float)
    if len(hist) < 2:
        return [float("nan")] * h

    if method == "naive":
        last = float(hist.iloc[-1])
        return [last] * h

    if method == "ma":
        if len(hist) < ma_window:
            return [float("nan")] * h
        v = float(hist.iloc[-ma_window:].mean())
        return [v] * h

    if method == "arima":
        try:
            from statsmodels.tsa.arima.model import ARIMA
        except Exception as e:
            raise RuntimeError(
                "ARIMA requires statsmodels. Install it (e.g., pip install statsmodels) "
                f"or choose --pred_method naive/ma. Original import error: {e}"
            )

        if len(hist) < max(30, sum(arima_order) + 10):
            return [float("nan")] * h

        model = ARIMA(hist.values, order=arima_order)
        fitted = model.fit()
        fc = fitted.forecast(steps=h)
        return [float(x) for x in fc]

    raise ValueError(f"Unknown method: {method}")


def score_from_forecasts(forecasts: List[float], last_open: float) -> float:
    if not np.isfinite(last_open) or last_open <= 0:
        return float("nan")
    vals = []
    for f in forecasts:
        if np.isfinite(f) and float(f) > 0:
            vals.append((float(f) / float(last_open)) - 1.0)
    return float(np.mean(vals)) if len(vals) > 0 else float("nan")


def simulate_holdout_topn_trading(
    universe: Dict[str, pd.DataFrame],
    holdout_n: int,
    context_n: int,
    initial_cash: float,
    top_n: int,
    pred_method: str,
    ma_window: int,
    arima_order: Tuple[int, int, int],
    horizon: int,
    min_score_threshold: float,
    rebalance_every: int,
    min_hold_days: int,
    max_buy_fraction: float,
    no_new_buys_last_day: bool,
    liquidate_end: bool = True,
) -> Dict[str, Any]:
    if holdout_n <= 0:
        raise ValueError("holdout_n must be >= 1")
    if context_n <= 0:
        raise ValueError("context_n must be >= 1")
    if top_n <= 0:
        raise ValueError("top_n must be >= 1")
    if horizon <= 0:
        raise ValueError("horizon must be >= 1")
    if rebalance_every <= 0:
        raise ValueError("rebalance_every must be >= 1")
    if min_hold_days < 0:
        raise ValueError("min_hold_days must be >= 0")
    if not (0 < max_buy_fraction <= 1.0):
        raise ValueError("max_buy_fraction must be in (0, 1].")

    dates = common_dates(universe)
    if len(dates) < (holdout_n + 3):
        raise ValueError(
            f"Not enough common dates across all tickers. "
            f"Need at least holdout_n+3 = {holdout_n+3}, got {len(dates)}"
        )

    start_hold_idx = len(dates) - holdout_n
    context_start_idx = max(0, start_hold_idx - context_n)

    U = {tic: df_by_date(df) for tic, df in universe.items()}

    portfolio = Portfolio(initial_cash=initial_cash)
    equity_rows = []
    daily_selection = []

    hold_age: Dict[str, int] = {}
    current_target: List[str] = []

    def _increment_hold_age():
        for tic in list(hold_age.keys()):
            if portfolio.holdings.get(tic, 0) > 0:
                hold_age[tic] = int(hold_age.get(tic, 0)) + 1
            else:
                hold_age.pop(tic, None)

    for idx in range(start_hold_idx, len(dates)):
        date_t = dates[idx]
        date_prev = dates[idx - 1]

        is_last_day = (idx == len(dates) - 1)
        do_rebalance = ((idx - start_hold_idx) % rebalance_every == 0)

        _increment_hold_age()

        selected: List[str] = current_target

        if do_rebalance:
            scores: List[Tuple[str, float]] = []
            for tic, dfi in U.items():
                if date_prev not in dfi.index or date_t not in dfi.index:
                    continue

                hist_open = dfi.loc[:date_prev, "Open"]
                last_open = float(dfi.loc[date_prev, "Open"])

                forecasts = predict_multi_step_opens(
                    history_open=hist_open,
                    method=pred_method,
                    ma_window=ma_window,
                    arima_order=arima_order,
                    horizon=horizon,
                )
                sc = score_from_forecasts(forecasts, last_open=last_open)
                if np.isfinite(sc) and sc >= float(min_score_threshold):
                    scores.append((tic, float(sc)))

            scores.sort(key=lambda x: x[1], reverse=True)
            selected = [tic for (tic, sc) in scores[: min(top_n, len(scores))]]
            current_target = selected

        daily_selection.append({"Date": date_t, "Selected": selected, "Rebalanced": int(do_rebalance)})

        close_prices_all = {tic: float(U[tic].loc[date_t, "Close"]) for tic in U.keys()}

        held_tickers = list(portfolio.holdings.keys())
        for tic in held_tickers:
            if portfolio.holdings.get(tic, 0) <= 0:
                continue
            if tic in selected:
                continue
            age = int(hold_age.get(tic, 0))
            if age < int(min_hold_days):
                continue
            px = float(U[tic].loc[date_t, "Open"])
            portfolio.sell_all(tic, px, date_t)
            hold_age.pop(tic, None)

        if (not (no_new_buys_last_day and is_last_day)) and len(selected) > 0:
            buy_list = [tic for tic in selected if portfolio.holdings.get(tic, 0) <= 0]
            if len(buy_list) > 0:
                cash = float(portfolio.cash)
                per_asset_budget = cash / float(len(buy_list)) if cash > 0 else 0.0
                per_asset_budget = min(per_asset_budget, cash * float(max_buy_fraction))

                for tic in buy_list:
                    px = float(U[tic].loc[date_t, "Open"])
                    if px <= 0 or per_asset_budget <= 0:
                        continue
                    shares = int(per_asset_budget // px)
                    if shares > 0:
                        portfolio.buy(tic, px, shares, date_t)
                        hold_age[tic] = 0

        portfolio.record_portfolio_value(close_prices_all, date_t)

        equity_rows.append({
            "Date": date_t,
            "NumSelected": len(selected),
            "Cash": portfolio.cash,
            "HoldingsValue": portfolio.get_holdings_value(close_prices_all),
            "TotalValue": portfolio.get_total_value(close_prices_all),
            "Rebalanced": int(do_rebalance),
        })

    if liquidate_end and len(equity_rows) > 0:
        last_date = equity_rows[-1]["Date"]
        held_tickers = list(portfolio.holdings.keys())
        for tic in held_tickers:
            if portfolio.holdings.get(tic, 0) > 0:
                px = float(U[tic].loc[last_date, "Open"])
                portfolio.sell_all(tic, px, last_date)
        hold_age.clear()

        close_prices_last = {tic: float(U[tic].loc[last_date, "Close"]) for tic in U.keys()}
        portfolio.value_history[-1]["cash"] = portfolio.cash
        portfolio.value_history[-1]["holdings_value"] = portfolio.get_holdings_value(close_prices_last)
        portfolio.value_history[-1]["total_value"] = portfolio.get_total_value(close_prices_last)

        equity_rows[-1]["Cash"] = portfolio.cash
        equity_rows[-1]["HoldingsValue"] = portfolio.get_holdings_value(close_prices_last)
        equity_rows[-1]["TotalValue"] = portfolio.get_total_value(close_prices_last)

    value_history = portfolio.get_value_history_df()
    transaction_history = portfolio.get_transaction_history_df()
    port_metrics = compute_metrics(value_history)

    return {
        "portfolio": portfolio,
        "equity_curve": pd.DataFrame(equity_rows),
        "value_history": value_history,
        "transaction_history": transaction_history,
        "portfolio_metrics": port_metrics,
        "daily_selection": pd.DataFrame(daily_selection),
        "dates_info": {
            "train_start": dates[0],
            "context_start": dates[context_start_idx],
            "holdout_start": dates[start_hold_idx],
            "holdout_end": dates[-1],
            "num_common_dates": len(dates),
        }
    }


def save_equity_and_sharpe_plot(eq: pd.DataFrame, out_dir: str, tag: str, sharpe_window: int = 5):
    if eq is None or eq.empty:
        return

    df = eq.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    df["ret"] = df["TotalValue"].pct_change()
    mu = df["ret"].rolling(sharpe_window).mean()
    sig = df["ret"].rolling(sharpe_window).std(ddof=0)
    df["rolling_sharpe"] = (mu / sig) * np.sqrt(252)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax1, ax2 = axes

    ax1.plot(df["Date"], df["TotalValue"], linewidth=1.8)
    ax1.set_title(f"Portfolio Value")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Value")
    ax1.grid(True, linewidth=0.3)

    ax2.plot(df["Date"], df["rolling_sharpe"], linewidth=1.8)
    ax2.set_title(f"Rolling Sharpe")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Sharpe")
    ax2.grid(True, linewidth=0.3)

    fmt = mdates.DateFormatter("%m-%d")
    ax1.xaxis.set_major_formatter(fmt)
    ax2.xaxis.set_major_formatter(fmt)
    fig.autofmt_xdate(rotation=0)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{tag}_equity_and_sharpe.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved plot -> {out_path}")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_dir", type=str, default="./data")
    ap.add_argument("--holdout_n", type=int, default=20)
    ap.add_argument("--context_n", type=int, default=80)
    ap.add_argument("--initial_cash", type=float, default=1_000_000.0)
    ap.add_argument("--top_n", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--min_score_threshold", type=float, default=0.0)
    ap.add_argument("--rebalance_every", type=int, default=3)
    ap.add_argument("--min_hold_days", type=int, default=2)
    ap.add_argument("--max_buy_fraction", type=float, default=0.2)
    ap.add_argument("--no_new_buys_last_day", action="store_true")
    ap.add_argument("--no_liquidate", action="store_true")
    ap.add_argument("--pred_method", type=str, default="ma", choices=["naive", "ma", "arima"])
    ap.add_argument("--ma_window", type=int, default=60)
    ap.add_argument("--arima_order", type=str, default="5,1,0")
    ap.add_argument("--out_dir", type=str, default="out_lab4_topN")
    ap.add_argument("--sharpe_window", type=int, default=5)
    args = ap.parse_args()

    try:
        p, d, q = [int(x.strip()) for x in args.arima_order.split(",")]
        arima_order = (p, d, q)
    except Exception:
        print("Invalid --arima_order. Use format p,d,q like 5,1,0")
        sys.exit(1)

    universe = build_universe(args.data_dir)
    print(f"[info] loaded tickers: {len(universe)}")

    try:
        res = simulate_holdout_topn_trading(
            universe=universe,
            holdout_n=args.holdout_n,
            context_n=args.context_n,
            initial_cash=args.initial_cash,
            top_n=args.top_n,
            pred_method=args.pred_method,
            ma_window=args.ma_window,
            arima_order=arima_order,
            horizon=args.horizon,
            min_score_threshold=args.min_score_threshold,
            rebalance_every=args.rebalance_every,
            min_hold_days=args.min_hold_days,
            max_buy_fraction=args.max_buy_fraction,
            no_new_buys_last_day=args.no_new_buys_last_day,
            liquidate_end=(not args.no_liquidate),
        )
    except RuntimeError as e:
        print(str(e))
        sys.exit(1)

    eq_df = res["equity_curve"]
    tr_df = res["transaction_history"]
    pm = res["portfolio_metrics"]
    sel_df = res["daily_selection"]
    info = res["dates_info"]

    tag = (
        f"TOP{args.top_n}_{args.pred_method}_OPEN_h{args.horizon}"
        f"_hold{args.holdout_n}_reb{args.rebalance_every}_minhold{args.min_hold_days}"
        f"_cap{args.max_buy_fraction}"
    )


    print("\n=== Strategy ===")
    print("Price alignment    : predict NEXT OPEN, trade at OPEN")
    print(f"Method             : {args.pred_method}")

    print("\n=== Portfolio Metrics (holdout trading) ===")
    print(f"Initial value     : ${pm['initial_value']:,.2f}")
    print(f"Final value       : ${pm['final_value']:,.2f}")
    print(f"Total return      : {pm['total_return']*100:,.3f}%")
    print(f"Annualized return : {pm['annualized_return']*100:,.3f}%")
    print(f"Sharpe ratio      : {pm['sharpe_ratio']:.4f}")
    print(f"Max drawdown      : {pm['max_drawdown']*100:,.3f}%")
    print(f"Trading days      : {pm['trading_days']}")

    print("\n=== Trades ===")
    if tr_df.empty:
        print("(no trades)")
    else:
        tmp = tr_df.reset_index().copy()
        cols = [c for c in ["date", "ticker", "action", "shares", "price", "cash_after", "cost", "revenue"] if c in tmp.columns]
        if "price" in tmp.columns:
            tmp["price"] = tmp["price"].map(lambda x: f"{float(x):,.2f}")
        if "cash_after" in tmp.columns:
            tmp["cash_after"] = tmp["cash_after"].map(lambda x: f"{float(x):,.2f}")
        if "cost" in tmp.columns:
            tmp["cost"] = tmp["cost"].map(lambda x: f"{float(x):,.2f}")
        if "revenue" in tmp.columns:
            tmp["revenue"] = tmp["revenue"].map(lambda x: f"{float(x):,.2f}")
        print(tmp[cols].to_string(index=False))

    os.makedirs(args.out_dir, exist_ok=True)

    save_equity_and_sharpe_plot(eq_df, out_dir=args.out_dir, tag=tag, sharpe_window=args.sharpe_window)

    eq_path = os.path.join(args.out_dir, f"{tag}_equity_curve.csv")
    tr_path = os.path.join(args.out_dir, f"{tag}_trades.csv")
    sel_path = os.path.join(args.out_dir, f"{tag}_daily_selection.csv")

    eq_df.to_csv(eq_path, index=False)
    tr_df.to_csv(tr_path)
    sel_df.to_csv(sel_path, index=False)

    print(f"\nSaved equity curve -> {eq_path}")
    print(f"Saved trades       -> {tr_path}")
    print(f"Saved selections   -> {sel_path}")


if __name__ == "__main__":
    main()
