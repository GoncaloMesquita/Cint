import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from statistics import stdev
def identify_outliers(data):

    print(data.Low.std())
    print(data.Low.mean())
    print(data.Low.describe())
    upper_limit = data.Low.mean() + 3*data['Low'].std()
    lower_limit = data.Low.mean() - 3*data['Low'].std()

    data_out = data[(data.Low > upper_limit) | (data.Low < lower_limit)]
    


#def remove_outliers():


df = pd.read_csv('EURUSD_Daily_Ask_2018.12.31_2019.10.05v2.csv', sep = ';', decimal=",")

df['Low'].plot()
plt.show()

Time = pd.to_datetime(df['Time (UTC)'])
startTime = datetime.datetime(2022, 9, 23, 0, 0)
identify_outliers(df)


    

# %%
