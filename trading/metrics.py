import numpy as np
import pandas as pd
import yfinance as yf

def total_return(value_history: pd.DataFrame):
    # Calculate total return as the percentage change from the initial value to the final value
    initial = value_history['total_value'].iloc[0]
    final = value_history['total_value'].iloc[-1]

    return (final - initial) / initial if initial != 0 else 0.0

def annualized_return(value_history: pd.DataFrame, trading_days_per_year: int = 252):
    # Note, 252 is the standard average number of trading days in a year, but this can be adjusted based on the actual data frequency
    initial = value_history['total_value'].iloc[0]
    final = value_history['total_value'].iloc[-1]
    n_days = len(value_history)

    if initial <= 0 or n_days <= 1:
        return 0.0
    
    return (final / initial) ** (trading_days_per_year / n_days) - 1

def sharpe_ratio(value_history: pd.DataFrame, risk_free_rate: float = 0.0360, trading_days_per_year: int = 252):
    # Calculate daily returns
    daily_returns = value_history['total_value'].pct_change().dropna()

    if daily_returns.empty:
        return 0.0
    
    if daily_returns.std() == 0:
        return 0.0
    
    daily_r = risk_free_rate / trading_days_per_year
    excess_returns = daily_returns - daily_r

    return np.sqrt(trading_days_per_year) * excess_returns.mean() / excess_returns.std()

def max_drawdown(value_history: pd.DataFrame):
    # Calculate the running maximum of the total value
    values = value_history['total_value']
    running_max = values.cummax()
    drawdown = (values - running_max) / running_max

    # Return the min drawdown because drawdowns are negative
    return drawdown.min()

def compute_metrics(value_history: pd.DataFrame, risk_free_rate: float = 0.0360, trading_days_per_year: int = 252):
    return {
        'total_return': total_return(value_history),
        'annualized_return': annualized_return(value_history, trading_days_per_year),
        'sharpe_ratio': sharpe_ratio(value_history, risk_free_rate, trading_days_per_year),
        'max_drawdown': max_drawdown(value_history),
        'initial_value': value_history['total_value'].iloc[0],
        'final_value': value_history['total_value'].iloc[-1],
        'trading_days': len(value_history)
    }