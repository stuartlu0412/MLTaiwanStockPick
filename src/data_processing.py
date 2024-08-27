import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

'''
    Data Processing
'''

def scale_data(data):
    
    returns_preserved = data['Seasonal Return'] #preseve return from any modification
    data_scaled = pd.DataFrame(scale(data), index = data.index, columns = data.columns) #scaling data
    data_scaled['Seasonal Return'] = returns_preserved #append back preserved return
    
    return data_scaled

def impute_data(data):
    
    returns_preserved = data['Seasonal Return'] #preseve return from any modification
    #imp = SimpleImputer(strategy='mean') #imputing data
    #imp.fit(data)
    #data_imputed = pd.DataFrame(imp.transform(data), index = data.index, columns = data.columns)
    data_imputed = data.groupby('證券代碼').ffill().fillna(0)
    data_imputed['Seasonal Return'] = returns_preserved #append back preserved return

    return data_imputed

def polynomial_transform(data, degree=1):

    returns_preserved = data['Seasonal Return'] #preseve return from any modification
    data.drop(columns=['Open', 'High', 'Low', 'Close', 'Volume', 'QuoteVolume',	'YSTD Close', 'TMR Close', 'Daily Return', 'YSTD Close Shift', 'Seasonal Return'], inplace=True)
    poly = PolynomialFeatures(interaction_only=False, degree = degree)
    poly.fit(data)
    data_transformed = pd.DataFrame(poly.transform(data), index = data.index, columns = poly.get_feature_names_out(data.columns))
    data_transformed['Seasonal Return'] = returns_preserved #append back preserved return

    return data_transformed

def train_test_split(data, split_date = '2022-01'):
    data_train = data[data.index.get_level_values(3) < split_date]
    data_test = data[data.index.get_level_values(3) >= split_date]
    return data_train, data_test