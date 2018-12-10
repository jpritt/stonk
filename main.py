# Machine learning classification libraries
from sklearn.svm import SVC
from sklearn.metrics import scorer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

# For data manipulation
import pandas as pd
import numpy as np
import math

# To plot
import matplotlib.pyplot as plt
import seaborn

# To fetch data
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
yf.pdr_override()

top_tech = ['AAPL', 'AMZN', 'MSFT', 'GOOG', 'FB', 'BABA', 'TCEHY', 'NFLX', 'EBAY', 'PYPL', 'BKNG', 'CRM', 'BIDU', 'JD', 'QTT']
penny_stocks = ['NM', 'RVLT', 'HSGX', 'TTNP', 'ESES', 'SNSS', 'GLG', 'NGD', 'LKM', 'DTRM', 'OHRP', 'AIPT']#, 'PLG', 'EGO', 'AAU', 'RRTS', 'ELGX', 'ESEA', 'UAVS', 'HEB', 'CBK', 'PGLC', 'CGIX', 'HUSA', 'CETX', 'AMCN', 'AUMN', 'IPWR', 'NVMM', 'GPL', 'TGB', 'SHIP', 'NXTD', 'BLIN', 'ALO', 'MICT', 'PTN', 'MTNB', 'THM', 'PLM', 'FLKS', 'ISR', 'PLX', 'EDGE', 'ABIO', 'ANFI', 'CPST', 'AKG', 'OESX', 'AGRX', 'SAUC']
companies = penny_stocks

# 1/1/12 is a Sunday

#from pandas_datareader.nasdaq_trader import get_nasdaq_symbols
#symbols = get_nasdaq_symbols()

data = pdr.get_data_yahoo(companies, "2012-01-01", "2018-12-07")
data.dropna()

dates = list(data.index)
days = (data.index[-1] - data.index[0]).days
weeks = math.ceil(days / 7.)
week_open, week_close, week_low, week_high, week_volume = [], [], [], [], []
X, y, pred = [], [], []
for c in companies:
    last_w = -1
    open = [0] * weeks
    close = [0] * weeks
    low = [float("inf")] * weeks
    high = [0] * weeks
    volume = [0] * weeks
    first_week = 0
    for i in range(len(dates)):
        w = int(((data.index[i] - data.index[0]).days + 1) / 7)
        if math.isnan(data.Open[c][i]):
            first_week = w+1
            continue
        if not w == last_w:
            open[w] = data.Open[c][i]
            if w > 0:
                close[w-1] = data.Close[c][i-1]
            last_w = w
        low[w] = min(low[w], data.Low[c][i])
        high[w] = max(high[w], data.High[c][i])
        volume[w] += data.Volume[c][i]
    close[-1] = data.Close[c][-1]

    week_open.append(open[first_week:])
    week_close.append(close[first_week:])
    week_high.append(high[first_week:])
    week_low.append(low[first_week:])
    week_volume.append(volume[first_week:])

    curr_X, curr_y = [], []
    for i in range(first_week+3,(weeks-1)):
        if close[i+1] / float(open[i]) > 1:
            curr_y.append(1)
        else:
            curr_y.append(0)

        vdelt1, vdelt2 = 0, 0
        if volume[i-1] > 0:
            vdelt1 = round(10*(volume[i]/float(volume[i-1])))
        if (volume[i-2]+volume[i-3]) > 0:
            vdelt2 = round(10*((volume[i]+volume[i-1])/float(volume[i-2]+volume[i-3])))
        v = [close[i]-open[i], close[i]-open[i-1], close[i]-open[i-3],\
             high[i]-low[i], high[i]-low[i-1], high[i]-low[i-3],\
             vdelt1, vdelt2]
        curr_X.append(v)
    X.append(curr_X)
    y.append(curr_y)

    pred.append([close[weeks-1]-open[weeks-1], close[weeks-1]-open[weeks-2], close[weeks-1]-open[weeks-4],\
             high[weeks-1]-low[weeks-1], high[weeks-1]-low[weeks-2], high[weeks-1]-low[weeks-4],\
             round(10*volume[weeks-1]/float(volume[weeks-2])),\
             round(10*(volume[weeks-1]+volume[weeks-2])/float(volume[weeks-3]+volume[weeks-4]))])

X_train, y_train = [], []
for i in range(len(companies)):
    size = len(week_open[i])
    split = int(0.8 * size)
    X_train += X[i][:split]
    y_train += y[i][:split]

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
cls = SVC(gamma='auto').fit(X_train, y_train)

for i in range(len(companies)):
    size = len(week_open[i])
    print('%s (%d weeks of data)' % (companies[i], size))
    split = int(0.8 * size)
    X_train, X_test = X[i][:split], X[i][split:]
    y_train, y_test = y[i][:split], y[i][split:]
    #cls = SVC(gamma='auto').fit(X_train, y_train)

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    accuracy_train = accuracy_score(y_train, cls.predict(X_train))
    accuracy_test = accuracy_score(y_test, cls.predict(X_test))
    print('Train Accuracy:{: .2f}%'.format(accuracy_train * 100))
    print('Test Accuracy:{: .2f}%'.format(accuracy_test * 100))

    p = cls.predict([pred[i]])
    if p[0]:
        print('Next week: BUY\n')
    else:
        print('Next week: SELL\n')
exit()

predicted = cls.predict(X)
start = 0
for i in range(len(data)):
    data[i]['Predicted_Signal'] = predicted[start:(start+len(data[i]))]
    split = int(split_percentage * len(data[i]))
    start += len(data[i])

    #if i == 0:
    #    plt.plot([data[i].index[split], data[i].index[-1]], [0,0], color='black', linewidth=1)

    # Calculate log returns
    data[i]['Return'] = np.log(data[i].Close.shift(-1) / data[i].Close) * 100
    data[i]['Strategy_Return'] = data[i].Return * data[i].Predicted_Signal
    data[i].Strategy_Return.iloc[split:].cumsum().plot(figsize=(15, 10), label=comps[i])
plt.grid(True)
plt.legend(loc='upper left')
plt.ylabel("Strategy Returns (%)")
plt.show()
