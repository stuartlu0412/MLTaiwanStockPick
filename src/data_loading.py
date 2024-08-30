import numpy as np
import pandas as pd

'''
    Data Loading
'''
def load_fundamental_data(path = './data/ifrs.csv'):
    '''
    This function loads seasonal fundamental data from CMoney.
    '''
    data = pd.read_csv(path, parse_dates=True, encoding='cp950')
    data.columns = data.columns.str.strip()
    data['年月'] = pd.to_datetime(data['年月'], format = '%Y%m').dt.to_period('M')
    data['財報發布日'] = pd.to_datetime(data['財報發布日'])
    data.set_index(['證券代碼', '財報發布日'], inplace=True)
    data.sort_index(level=0, inplace=True)
    data.drop(columns=['合併(Y/N)', '單季(Q)/單半年(H)', '月份', '季別', '幣別', '市場別', '財報附註TEJ是否完成Y/N', 
                       '財報類別（1個別2個體3合併）', '財報年月起日', '財報年月迄日', '市場別','財報附註TEJ是否完成Y/N',], inplace = True)
    data.dropna(axis=1, how='all', inplace= True)
    return data

def load_price_data(path = './data/price_daily.csv'):
    '''
    This function loads daily price data from CMoney
    '''
    price = pd.read_csv(path, encoding='cp950')
    price.columns = price.columns.str.strip()
    price['年月日'] = pd.to_datetime(price['年月日'], format = '%Y%m%d')
    price.set_index(['證券代碼', '年月日'], inplace=True)
    price = price.rename(columns={'開盤價(元)': 'Open',
                          '最高價(元)': 'High',
                          '最低價(元)': 'Low',
                          '收盤價(元)': 'Close',
                          '成交量(千股)': 'Volume',
                          '成交值(千元)': 'QuoteVolume',
                          '市值(百萬元)': 'MarketCap'})
    price = price.sort_index(level=0)
    price = price.dropna(axis=1, how='all')
    price = price[['Open', 'High', 'Low', 'Close', 'Volume', 'QuoteVolume', 'MarketCap']]

    return price

def combine_df(*dfs):
    new_df = dfs[0]
    for i in range(1, len(dfs)):
        new_df = pd.concat((new_df, dfs[i]), axis=0)
    new_df.drop_duplicates(inplace=True)
    new_df.sort_index(level='證券代碼', inplace=True)
    return new_df

def calculate_daily_return(price):
    
    price['YSTD Close'] = price.groupby('證券代碼')['Close'].shift(1)
    price['TMR Close'] = price.groupby('證券代碼')['Close'].shift(-1)
    price['Daily Return'] = (price['TMR Close'] / price['Close']).apply(np.log)
    price = price[['Open', 'High', 'Low', 'Close', 'Volume', 'QuoteVolume', 'MarketCap', 'YSTD Close', 'TMR Close', 'Daily Return']]
    return price

def remove_special_stocks(price):
    special_stocks = ['3702A 大聯大甲特',
                    '2891B 中信金乙特',
                    '2891C 中信金丙特',
                    '5871A 中租-KY甲特',
                    '2002A 中鋼特',
                    '3036A 文曄甲特',
                    '2897A 王道銀甲特',
                    '1101B 台泥乙特',
                    '2887Z1 台新己特',
                    '2887E 台新戊特',
                    '2887F 台新戊特二',
                    '8112A 至上甲特',
                    '6592B 和潤企業乙特',
                    '6592A 和潤企業甲特',
                    '8349A 恒耀甲特',   
                    '2348A 海悅甲特',
                    '2836A 高雄銀甲特',
                    '2882B 國泰金乙特',
                    '2882A 國泰特',
                    '1312A 國喬特',
                    '1522A 堤維西甲特',
                    '2881B 富邦金乙特',
                    '2881C 富邦金丙特',
                    '2881A 富邦特',
                    '2883B 開發金乙特',
                    '2888B 新光金乙特',
                    '2888A 新光金甲特',
                    '9941A 裕融甲特',
                    '2838A 聯邦銀甲特']
    exclude_stocks = price.index.get_level_values('證券代碼').isin(special_stocks)
    price = price[~exclude_stocks]
    return price

def fundamental_data_drop_duplicate(data):
    data.reset_index(inplace=True)
    data.drop_duplicates(subset=['證券代碼', '年月'], inplace=True, keep='first')
    data.set_index(['證券代碼', '財報發布日'], inplace=True)
    return data

def select_features(data, features, rename):
    data = data.rename(columns = dict(zip(features, rename)))
    return data[rename]

def merge_feature_data(feature: pd.DataFrame, price, return_mode = 'simple'):
    '''
    This function merge data with different frequency and return seasonal data 
    that contain entry and exit for each period.
    '''
    #merge data and convert frequency to seasonal
    data_merged = feature.merge(price, how='left', left_index=True, right_on=['證券代碼', '年月日'])
    #seasonal = (data_merged.index.get_level_values('財報發布日') == data_merged.index.get_level_values('年月日'))
    #data_seasonal = data_merged[seasonal]
    data_seasonal = data_merged
    
    data_seasonal.sort_index(level = '年月日')
    #calculate returns
    data_seasonal['YSTD Close Shift'] = data_seasonal.groupby('證券代碼')['YSTD Close'].shift(-1)
    
    if return_mode == 'simple':
        data_seasonal['Seasonal Return'] = (data_seasonal['YSTD Close Shift'] / data_seasonal['TMR Close']) - 1
    else:
        data_seasonal['Seasonal Return'] = (data_seasonal['YSTD Close Shift'] / data_seasonal['TMR Close']).apply(np.log)

    data_seasonal.set_index('年月', append=True, inplace=True)
    data_seasonal.sort_index(level='年月', inplace = True)
    
    return data_seasonal