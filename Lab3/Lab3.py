
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from statistics import stdev

def identify_outliers(data):

    #print(data.Low.describe())
    upper_limit = data.Low.mean() + 3*data['Low'].std()
    lower_limit = data.Low.mean() - 3*data['Low'].std()

    out = data[(data.Low > upper_limit) | (data.Low < lower_limit)] 
    return out

def taking_outliers(data, outlier):

    data_clean = data['Low'].drop(outlier.index[0], inplace = False)
    data_clean.plot()
    plt.show()
    return

def define_outlier_previous(data, outlier):

    data['Low'] = data['Low'].replace(outlier.index[0], data['Low'][outlier.index[0]-1])
    print(data['Low'][100:110])
#EXERCISE 1

df = pd.read_csv('EURUSD_Daily_Ask_2018.12.31_2019.10.05v2.csv', sep = ';', decimal=",")

df.plot()
plt.show()

Time = pd.to_datetime(df['Time (UTC)'])
startTime = datetime.datetime(2022, 9, 23, 0, 0)
outliers = identify_outliers(df)
taking_outliers(df, outliers)
define_outlier_previous(df, outliers)

#EXERCISE 2

df1 = pd.read_csv('DCOILBRENTEUv2.csv')
df1.DATE = pd.to_datetime(df1.DATE)

df1.plot(x = 'DATE')
plt.show()

df1['DCOILBRENTEU'].diff().hist(bins=35)
plt.show()

df1['DCOILBRENTEU'].groupby(df1['DATE'].dt.year).count().plot(kind="bar")
plt.show()

#EXERCICE 3 

df2 = pd.read_csv('DCOILWTICOv2.csv')
print(df2)

ax = df1['DCOILBRENTEU'].plot()

df2['DCOILWTICO'].plot(ax=ax)
plt.show()