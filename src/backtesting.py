import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

'''
Backtesting and plot portfolio
'''

def backtest_daily_stocksort(data, price, clf, q=10, market_cap_lb=0, market_cap_ub=np.inf):

    history = data.copy()
    history['Probability'] = clf.predict_proba(history.drop(columns=['Seasonal Return', 'Cross Section Median']))[:, 1]
    history = history.reset_index()
    history.set_index(['證券代碼', '年月日'], inplace=True)

    # Merge with price data and handle missing values
    history_merged = price[['MarketCap', 'Daily Return']].merge(history[['Probability']], left_index=True, right_index=True, how='outer')
    preserved_return = history_merged['Daily Return']
    history_merged = history_merged.groupby('證券代碼').ffill()
    history_merged['Daily Return'] = preserved_return#.groupby('證券代碼').shift(-1)  # Shift return to avoid look-ahead bias
    history_merged.reset_index(inplace=True)

    history_merged['Probability'] = history_merged.groupby('證券代碼')['Probability'].shift(1)
    history_merged.sort_values(by='年月日')
    history_merged.dropna(how='any', inplace=True)

    # Filter stocks based on market cap and sort by date
    filtered_stocks = history_merged[(history_merged['MarketCap'] >= market_cap_lb) & (history_merged['MarketCap'] < market_cap_ub)]
    filtered_stocks.sort_values(by=['年月日', 'Probability'], ascending=[True, False], inplace=True)

    # Assign quantiles
    filtered_stocks['Quantile'] = filtered_stocks.groupby('年月日')['Probability'].transform(
        lambda x: pd.qcut(x.rank(method='first'), q, labels=False, duplicates='drop'))

    # Calculate mean returns for each quantile and date
    quantile_returns = filtered_stocks.groupby(['年月日', 'Quantile'])['Daily Return'].mean().unstack(level='Quantile')

    filtered_stocks['Weight'] = 1 / filtered_stocks.groupby(['年月日', 'Quantile']).transform('size')
    filtered_stocks.set_index(['年月日', '證券代碼'], inplace=True)

    # Calculate position for each quantile
    position = pd.DataFrame()

    for i in range(q):
        position[i] = filtered_stocks['Weight'].where(filtered_stocks['Quantile'] == i, other=0)

    return quantile_returns, position

def plot_portfolio(portfolio, long = 0, short = 0, tc = 0.0001, compounded = False):
    n = len(portfolio.columns)
    if not long:
        long = n - 1
    if not short:
        short = 0
    portfolio['long-short'] = (portfolio[long] - portfolio[short]) - tc

    now = datetime.now()

    benchmark = yf.download('^TWII', start = '2015-05-15', end = now)
    tw_0050 = yf.download('0050.TW', start = '2015-05-15', end = now)
    twf = yf.download('00632R.TW', start = '2015-05-15', end = now)
    portfolio['benchmark'] = np.log(benchmark['Adj Close'] / benchmark['Adj Close'].shift(1)).shift(-1)
    portfolio['0050'] = np.log(tw_0050['Adj Close'] / tw_0050['Adj Close'].shift(1)).shift(-1)
    portfolio['reverse'] = np.log(twf['Adj Close'] / twf['Adj Close'].shift(1)).shift(-1)
    
    for i in range(n):
        portfolio[i] = portfolio[i] - tc

    if compounded:
        ((portfolio[['long-short', short, long, 'benchmark', '0050']]+1).cumprod()).plot()
    else:
        ((portfolio[['long-short', short, long, 'benchmark', '0050']]).cumsum() + 1).plot()

def show_position_at_given_time(position, quantile, date, save=False):
    pos = position[quantile][position[quantile] > 0][date]
    if save:
        pos.to_csv('position.csv', encoding='cp950')
    return pos

def show_position_for_given_stock(position, quantile, stock = '2882 國泰金'):
    
    pos = position[[quantile]][position.index.get_level_values(level=1) == stock]
    pos.reset_index(inplace=True)
    pos.drop(columns=['證券代碼'], inplace=True)
    pos.set_index('年月日', inplace=True)
    pos = pos[9]
    pos.plot()
    return pos