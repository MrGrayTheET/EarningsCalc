import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import pandas_datareader as pdr
from keras.layers import Dense, Dropout,LSTM, GRU
from keras.models import Sequential
import pandas_ta as ta
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import  adfuller
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

start = dt.datetime(1991,1,1)
end = dt.datetime(2023,2,1)
ffr = pdr.DataReader('DFF', 'fred', start=start, end=end)
ppi = pdr.DataReader('PPIACO', 'fred', start=start,end=end)
earnings = pdr.DataReader('BOGZ1FA106110115Q','fred', start=start,end=end)
unem = pdr.DataReader('UNRATE', 'fred', start=start, end=end)
sent = pdr.DataReader('UMCSENT', 'fred', start=start, end=end)
mfo = pdr.DataReader('AMTMNO', 'fred', start=start, end=end)
mfo['AMTMNO'].rename('orders')

ffr = ffr.resample('M').last()

sales = pdr.DataReader('CMRMTSPL', 'fred', start=start, end=end)
sales['CMRMTSPL'].rename('sales')

df = pd.concat([earnings, sales,sent,ffr],1).fillna(method='ffill').reindex(earnings.index)
df = df.dropna().reset_index()

df = df.drop('DATE', 1)

def adfuller_test(series, signif=0.05, name='', verbose=False):
    """Perform ADFuller to test for Stationarity of given series and print report"""
    r = adfuller(series, autolag='AIC')
    output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}
    p_value = output['pvalue']
    def adjust(val, length= 6): return str(val).ljust(length)

    # Print Summary
    print(f'    Augmented Dickey-Fuller Test on "{name}"', "\n   ", '-'*47)
    print(f' Null Hypothesis: Data has unit root. Non-Stationary.')
    print(f' Significance Level    = {signif}')
    print(f' Test Statistic        = {output["test_statistic"]}')
    print(f' No. Lags Chosen       = {output["n_lags"]}')

    for key,val in r[4].items():
        print(f' Critical value {adjust(key)} = {round(val, 3)}')

    if p_value <= signif:
        print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
        print(f" => Series is Stationary.")
    else:
        print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")
        print(f" => Series is Non-Stationary.")

def format_data(scaled_data, past_periods, future_periods):
    trainX = []
    trainY = []
    for i in range(past_periods, len(scaled_data) - future_periods + 1):
        trainX.append(scaled_data[i - past_periods:i, 0:scaled_data.shape[1]])
        trainY.append(scaled_data[i + future_periods - 1:i + future_periods, 0])

    #
    trainX, trainY = np.array(trainX), np.array(trainY)
    return trainX, trainY

def CreateLSTMBase(trainX, trainY):
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True)) #Return from one cell to the next
    model.add(LSTM(32, activation='relu', return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(trainY.shape[1]))
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    model.save('modelh1')
    return  model

def CreateGRUBase(trainX, trainY):
    model = Sequential()
    model.add(GRU(64,activation='relu',input_shape=(trainX.shape[1], trainX.shape[2]),return_sequences=True))
    model.add(GRU(32, activation='relu'))
    model.add(Dense(trainY.shape[1]))
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    model.save('modelh2')
    return model
# Preprocessing

scaler =StandardScaler()
scaler = scaler.fit(df)
dfScaled = scaler.transform(df) # Scale our data for standardization
trainLen = int(0.7 * len(dfScaled))
training_data = dfScaled[:trainLen]
testing_data = dfScaled[trainLen:]
trainX,trainY = format_data(training_data, 20, 8)
testX, testY = format_data(testing_data, 20, 8)



modelG = CreateGRUBase(trainX, trainY)
modelL = CreateLSTMBase(trainX, trainY)
history = modelL.fit(trainX, trainY, epochs=50, batch_size=20,validation_split=0.1, verbose=1)
history2 = modelG.fit(trainX,trainY,batch_size=20, epochs=50, validation_split=0.1, verbose=1)
fig, ax = plt.subplots()
ax.plot(history.history['loss'])
pred = modelG.predict(testX)
p