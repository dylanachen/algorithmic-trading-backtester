import pandas as pd
from datetime import datetime

class Portfolio:
    def __init__(self, initial_cash):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        # Dictionary with key: ticker symbol, value: number of shares held
        self.holdings = {}
        self.transaction_history = []
        self.value_history = []

    def buy(self, ticker: str, price: float, shares: int, date=None):
        # Buying shares of a stock from the ticker symbol (date optional)
        cost = price * shares

        # Return false if there are not sufficient funds to buy the shares requested
        if cost > self.cash:
            print(f"Insufficient funds to buy {shares} shares of {ticker} at ${price:.2f} each.")
            return False
        
        self.cash -= cost
        self.holdings[ticker] = self.holdings.get(ticker, 0) + shares

        # Record the transaction
        self.transaction_history.append({
            'date': date,
            'ticker': ticker,
            'action': "BUY",
            'shares': shares,
            'price': price,
            'cost': cost,
            'cash_after': self.cash
        })

        return True

    def sell(self, ticker: str, price: float, shares: int, date=None):
        # Selling shares of a stock from the ticker symbol (date optional)
        revenue = price * shares

        # Return false if there are not sufficient shares to sell the shares requested
        if self.holdings.get(ticker, 0) < shares:
            print(f"Not enough shares to sell {shares} of {ticker}.")
            return False

        self.cash += revenue
        self.holdings[ticker] -= shares

        # Remove the ticker from holdings if all shares are sold
        if self.holdings[ticker] == 0:
            del self.holdings[ticker]
        
        # Record the transaction
        self.transaction_history.append({
            'date': date,
            'ticker': ticker,
            'action': "SELL",
            'shares': shares,
            'price': price,
            'revenue': revenue,
            'cash_after': self.cash
        })

        return True

    def buy_max(self, ticker: str, price: float, date=None):
        # Buying the maximum number of shares possible with the available cash
        shares = int(self.cash // price)
        return self.buy(ticker, price, shares, date)

    def sell_all(self, ticker: str, price: float, date=None):
        # Selling all shares of a stock from the ticker symbol
        shares = self.holdings.get(ticker, 0)
        return self.sell(ticker, price, shares, date)
    
    def get_holdings_value(self, current_prices: dict):
        # Calculate the total value of the current holdings based on the provided price dictionary
        value = 0.0

        for ticker, shares in self.holdings.items():
            if ticker in current_prices:
                price = current_prices[ticker]
                value += shares * price
        return value
    
    def get_total_value(self, current_prices: dict):
        # Calculate the total value of the portfolio (cash + holdings)
        holdings_value = self.get_holdings_value(current_prices)
        total_value = self.cash + holdings_value
        return total_value

    def record_portfolio_value(self, current_prices: dict, date=None):
        # Record the total value of the portfolio at a given date
        holdings_value = self.get_holdings_value(current_prices)

        self.value_history.append({
            'date': date,
            'cash': self.cash,
            'holdings_value': holdings_value,
            'total_value': self.cash + holdings_value
        })
    
    def get_value_history_df(self):
        # Return the value history as a pandas DataFrame
        df = pd.DataFrame(self.value_history)
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        return df

    def get_transaction_history_df(self):
        # Return the transaction history as a pandas DataFrame
        df = pd.DataFrame(self.transaction_history)
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        return df
