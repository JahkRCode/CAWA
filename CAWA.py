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
#plt.plot(daily_cases)
#plt.title('Current Total Daily Cases')

daily_cases = daily_cases.diff().fillna(daily_cases[0]).astype(np.int64)
print(daily_cases.head())
#plt.plot(daily_cases)
#plt.title('Daily Cases')
print(f'DAILY CASE SHAPE : {daily_cases.shape}')

## ***** Preprocessing ***** ##

test_data_size = 7 ## Number of days to test

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
    
    
seq_length = 5
X_train, y_train = prepare_sequences(train_data, seq_length)
X_test, y_test = prepare_sequences(test_data, seq_length)

X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()

X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()
    

## ***** Building a Model ***** ##

class CAWAModel(nn.Module):
    
    ## Initialize variables for model
    ## n_features -->
    ## n_hidden -->
    ## seq_len -->
    ## n_layers -->
    
    def __init__(self, n_features, n_hidden, seq_len, n_layers=2):
        super(CAWAModel, self).__init__()
        
        self.n_hidden = n_hidden
        self.seq_len = seq_len
        self.n_layers = n_layers
        
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=n_hidden,
            num_layers=n_layers,
            dropout=0.5
        )
        self.linear = nn.Linear(in_features=n_hidden, out_features=1)
    
    ## For a stateless LSTM this method resets the states after each example
    def reset_hidden_state(self):
        self.hidden = (
            torch.zeros(self.n_layers, self.seq_len, self.n_hidden),
            torch.zeros(self.n_layers, self.seq_len, self.n_hidden)
        )
        
    ## 
    def forward(self, sequences):
        lstm_out, self.hidden = self.lstm(
            sequences.view(len(sequences), self.seq_len, -1),
            self.hidden
        )
        last_time_step = \
            lstm_out.view(self.seq_len, len(sequences), self.n_hidden)[-1]
        y_pred = self.linear(last_time_step)
        
        return y_pred
        
def train_model(
        model,
        train_data,
        train_labels,
        test_data=None,
        test_labels=None
    ):
    loss_fn = torch.nn.MSELoss(reduction='sum')
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 100
    
    train_hist = np.zeros(num_epochs)
    test_hist = np.zeros(num_epochs)
    
    for epoch in range(num_epochs):
        model.reset_hidden_state()
        
        y_pred = model(X_train)
        
        loss = loss_fn(y_pred.float(), y_train)
        
        if test_data is not None:
            with torch.no_grad():
                y_test_pred = model(X_test)
                test_loss = loss_fn(y_test_pred.float(), y_test)
            test_hist[epoch] = test_loss.item()
             
            if epoch % 10 == 0:
                print(f'Epoch {epoch} train loss: {loss.item()}' \
                      'test loss: {test_loss.item()}')
                    
        elif epoch % 10 == 0:
            print(f'Epoch {epoch} train loss: {loss.item()}')        
        
        train_hist[epoch] = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return model.eval(), train_hist, test_hist
        
model = CAWAModel(
    n_features=1,
    n_hidden=512,
    seq_len=seq_length,
    n_layers=4
)       
model, train_hist, test_hist = train_model(
    model,
    X_train,
    y_train,
    X_test,
    y_test
)

with torch.no_grad():
    test_seq = X_test[:1]
    preds = []
    for _ in range(len(X_test)):
        y_test_pred = model(test_seq)
        pred = torch.flatten(y_test_pred).item()
        preds.append(pred)
        new_seq = test_seq.numpy().flatten()
        new_seq = np.append(new_seq, [pred])
        new_seq = new_seq[1:]
        test_seq = torch.as_tensor(new_seq).view(1, seq_length, 1).float()

true_cases = scaler.inverse_transform(
    np.expand_dims(y_test.flatten().numpy(), axis=0)
).flatten()

predicted_cases = scaler.inverse_transform(
    np.expand_dims(preds, axis=0)
).flatten()

print(f'PREDICTTED CASES: {predicted_cases}')
print(f'ACTUAL CASES: {true_cases}')
## ***** Visualize Results ***** ##

plt.plot(
    daily_cases.index[:len(train_data)],
    scaler.inverse_transform(train_data).flatten(),
    label = 'Historial Daily Cases'    
)
plt.plot(
    daily_cases.index[len(train_data):len(train_data) + len(true_cases)],
    true_cases,
    label = 'Real Daily Cases'
)
plt.plot(
    daily_cases.index[len(train_data):len(train_data) + len(true_cases)],
    predicted_cases,
    label = 'Predicted Daily Cases'    
)

plt.legend();







## ***** End of Code ***** ##
