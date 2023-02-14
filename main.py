import numpy as np
import pandas as pd
import datetime as dt
import pandas_datareader as pdr
from keras.layers import Dense, Dropout,LSTM, GRU
from keras.models import Sequential

import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import  adfuller
from sklearn.preprocessing import StandardScaler


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

#Build our functions
def format_data(scaled_data, past_periods, future_periods): #Past Periods = Lookback
                                                            #Future periods = values to be predicted
    trainX = []
    trainY = []
    for i in range(past_periods, len(scaled_data) - future_periods + 1):
        trainX.append(scaled_data[i - past_periods:i, 0:scaled_data.shape[1]])
        trainY.append(scaled_data[i + future_periods - 1:i + future_periods, 0]) #Shaping data for multivariate analysis

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
training_data = dfScaled[:trainLen] #Split our data for train/test
testing_data = dfScaled[trainLen:]

trainX,trainY = format_data(training_data, 20, 8) # Define our data, lookback, future
testX, testY = format_data(testing_data, 20, 8)



modelG = CreateGRUBase(trainX, trainY) #Building the GRU
modelL = CreateLSTMBase(trainX, trainY) #Building LSTM
history = modelL.fit(trainX, trainY, epochs=50, batch_size=20,validation_split=0.1, verbose=1)
history2 = modelG.fit(trainX,trainY,batch_size=20, epochs=50, validation_split=0.1, verbose=1) #Fitting
fig, ax = plt.subplots()
ax2 = ax.twinx()
ax.plot(history.history['loss']) #LSTM LOSS
ax2.plot(history2.history['loss'], color='orange') #GRU LOSS

pred = modelL.predict(testX)
pred2 = modelG.predict(testX)

pred_copy = np.repeat(pred, dfScaled.shape[1], -1)
pred2_copy = np.repeat(pred2, dfScaled.shape[1], -1) #Build our forecast-copies
trans_pred = scaler.inverse_transform(pred_copy)[:, 0]
trans_pred2 = scaler.inverse_transform(pred2_copy)[:, 0]

fig, pplot = plt.subplots()
pplot2 = pplot.twinx()
pplot.plot(trans_pred)
pplot2.plot(trans_pred2, color='orange')
plt.show()