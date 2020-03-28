#######################################################
## Name: JahkRCode
## Title: CAWA.py
## Description: CAWA -> Coronavirus Analysis Worldwide Assessment
## Date Created: March 27, 2020
#######################################################
import torch

import os
import numpy as np
from tqdm import tqdm
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.preprocessing import MinMaxScaler
from pandas.plotting import register_matplotlib_converters
import pandas as pd
from torch import nn, optim
import torch.nn.functional as F

#%rc inline
#%config InlineBackend.figure_format = 'retina'

sns.set(style='whitegrid', palette='muted', font_scale=1.2)

HAPPY_COLORS_PALETTE = ['#01BEFE', '#FFDD00', '#FF7D00', '#FF006D', '#93D30C','#8F00FF']

sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

rcParams['figure.figsize'] = 12, 6
register_matplotlib_converters()

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

df = pd.read_csv('time_series_covid19_confirmed_global.csv')
print(df.head())

df = df.iloc[:, 4:]
print(df.head())

daily_cases = df.sum(axis=0)
daily_cases.index = pd.to_datetime(daily_cases.index)
print(daily_cases.head())
plt.plot(daily_cases)
plt.title('Current Total Daily Cases')

daily_cases = daily_cases.diff().fillna(daily_cases[0]).astype(np.int64)
print(daily_cases.head())
plt.plot(daily_cases)
plt.title('Daily Cases')
print(f'DAILY CASE SHAPE : {daily_cases.shape}')

## ***** Preprocessing ***** ##

test_data_size = 14 ## Number of days to test

train_data = daily_cases[:-test_data_size]
test_data = daily_cases[-test_data_size:]

print(f'TRAINIG DATA SHAPE : {train_data.shape}')
print(f'TESTING DATA SHAPE : {test_data.shape}')

scaler = MinMaxScaler()
scaler = scaler.fit(np.expand_dims(train_data, axis=1))

train_data = scaler.transform(np.expand_dims(train_data, axis=1))
test_data = scaler.transform(np.expand_dims(test_data, axis=1))

def prepare_sequences(data, seq_length):
    xs = []
    ys = []
    
    for i in range(len(data) - seq_length - 1):
        x = data[i:(i+seq_length)]
        y = data[(i+seq_length)]
        xs.append(x)
        ys.append(y)
        
    return np.array(xs), np.array(ys)
    
    
    
    
    
    
    

        
        
        
        
        
        
        
        



## ***** End of Code ***** ##
