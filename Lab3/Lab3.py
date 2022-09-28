
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from statistics import stdev

def identify_outliers(data, coll):

    #print(data.Low.describe())
    upper_limit = data[coll].mean() + 3*data[coll].std()
    lower_limit = data[coll].mean() - 3*data[coll].std()

    out = data[coll][(data[coll] > upper_limit) | (data[coll] < lower_limit)] 
    return out

def taking_outliers(data, outlier, coll):

    data_clean = data[coll].drop(outlier.index[0], inplace = False)
    data_clean.plot()
    plt.title(coll)
    plt.show()

    return

def define_outlier_previous(data, outlier, coll):
    
    data.iloc[outlier.index, data.columns == coll ] = data.iloc[outlier.index - 1, data.columns == coll ]

    print(data[coll][100:110])
    return 
#EXERCISE 1

df = pd.read_csv('EURUSD_Daily_Ask_2018.12.31_2019.10.05v2.csv', sep = ';', decimal=",")

df = df.rename(columns={"Volume ": "Volume"})
df['Time (UTC)'] = pd.to_datetime(df['Time (UTC)'])

plt.plot(df['Time (UTC)'],df['Open'])
plt.plot(df['Time (UTC)'],df['Low'])
plt.show()
plt.plot(df['Time (UTC)'],df['Volume'])
plt.show() 


startTime = datetime.datetime(2022, 9, 23, 0, 0)
j= 0
for i in df.columns:
    if j >=1:
        outliers = identify_outliers(df, i)
        taking_outliers(df, outliers, i)
        define_outlier_previous(df, outliers, i)
    j=j+1
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

plt.plot(df1['DCOILBRENTEU'])
plt.plot(df2['DCOILWTICO'])
plt.show() 