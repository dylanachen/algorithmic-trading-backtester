import pandas as pd
from trading.portfolio import Portfolio

def run_backtest(price_data: pd.DataFrame, signals, initial_cash: float = 100000, allocation: str = "equal", trade_percentage: float = 0.1):
    '''
    Runs a backtest over historical price data using the provided trading signals
    
    Parameters:
    - price_data: DataFrame with datetime index and one column per ticker containing historical price data

    - signals: NEEDS TO BE DEFINED: a callable that takes the historical price data up until present day and returns a dictionary of ticker symbols to buy, sell, or hold
    
    - initial_cash: Starting cash for the portfolio
    - allocation: How to allocate cash across tickers with buy signals
    - trade_percentage: Percentage of available cash to use for each trade
    
    Returns:
    - portfolio: The final state of the Portfolio after running the backtest
    - value_history: DataFrame of the portfolio's total value over time
    '''
    # Get the list of tickers from the price data columns and initialize the portfolio
    tickers = price_data.columns.tolist()
    portfolio = Portfolio(initial_cash=initial_cash)

    # Loop through each date in the price data
    for i in range(len(price_data)):
        date = price_data.index[i]
        current_prices = price_data.iloc[i].to_dict()

        # Get signals from model for historical price data up until the current date
        current_history = price_data.iloc[:i+1]
        signals = signals(current_history)

        # Perform trades
        buy_tickers = signals.get('buy', [])
        sell_tickers = signals.get('sell', [])
        hold_tickers = signals.get('hold', [])

        # Sell first to free up cash for buys
        if sell_tickers:
            for ticker in sell_tickers:
                if ticker in current_prices:
                    price = current_prices[ticker]
                    portfolio.sell_all(ticker, price, date)

        # Then buy with the available cash
        if buy_tickers:
            if allocation == "equal":
                cash_per_ticker = portfolio.cash * trade_percentage / len(buy_tickers)
            else:
                cash_per_ticker = portfolio.cash * trade_percentage
            
            for ticker in buy_tickers:
                if ticker in current_prices:
                    price = current_prices[ticker]
                    portfolio.buy(ticker, price, cash_per_ticker // price, date)
            
        # Record the portfolio value at the end of the day
        portfolio.record_portfolio_value(current_prices, date)

    # Get the value history as a DataFrame
    value_history = portfolio.get_value_history_df()
    transaction_history = portfolio.get_transaction_history_df()

    final_portfolio_value = value_history['total_value'].iloc[-1] if not value_history.empty else initial_cash
    final_prices = price_data.iloc[-1].to_dict() if not price_data.empty else {}

    results = {
        'initial_cash': initial_cash,
        'final_portfolio_value': final_portfolio_value,
        'final_holdings_value': portfolio.get_holdings_value(final_prices),
        'final_cash': portfolio.cash,
        'value_history': value_history,
        'transaction_history': transaction_history
    }

    return portfolio, results