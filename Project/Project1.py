import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np


#---------------------Functions------------------------------

def data_visualization(data, data_clean, coll):
    #print(df.describe())
    """ data[coll].plot()
    plt.title(coll)
    plt.show() """
    fig, axs = plt.subplots(2)
    axs[0].plot(data[coll])
    axs[1].plot(data_clean[coll])
    plt.title(coll)
    plt.show()

    return

def identify_outliers1(data,  coll):

    upper_limit = data[coll].mean() + 4*data[coll].std()
    lower_limit = data[coll].mean() - 4*data[coll].std()

    out = data[coll][(data[coll] > upper_limit) | (data[coll] < lower_limit)] 
    miss_values = data[data[coll].isnull()]
    return out, miss_values

""" def identify_outliers2(data, coll):
    q1 = df[coll].quantile(0.25)
    q3 = df[coll].quantile(0.75)
    iqr = q3 - q1
    lower_limit = df[coll].quantile(0.03) -(1.5 * iqr) 
    upper_limit = df[coll].quantile(.97) +(1.5 * iqr)
    out = data[coll][(data[coll] > upper_limit) | (data[coll] < lower_limit)] 
    return out """

def replacing_outliers_missvalues(data, outlier,miss_value, coll):
    
    #data.iloc[outlier.index, data.columns == coll ] = (data.iloc[outlier.index - 1, data.columns == coll ] + data.iloc[outlier.index +1, data.columns == coll ])/2
    data.iloc[outlier.index, data.columns == coll ] = data.iloc[outlier.index - 1, data.columns == coll ] 
    data.iloc[miss_value.index, data.columns == coll ] = data.iloc[miss_value.index - 1, data.columns == coll ] 
    #data.iloc[outlier.index, data.columns == coll ] = np.nan
    #data.interpolate(method = 'linear')
    return data
#---------------------MAIN-----------------------------------

df = pd.read_csv('Proj1_Dataset.csv')
#df['Time'] = pd.to_datetime(df['Time'])
#df['Date'] = pd.to_datetime(df['Date'])

#print(df.info())
#print(df.describe())
#print(df.isnull().sum()) 

print(df.describe())
df_clean = df.copy()

for i in df.columns:
    
    if i != "Time" and i !="Date":  
        outliers, missing_values = identify_outliers1(df_clean, i)
        df_clean = replacing_outliers_missvalues(df_clean, outliers, missing_values, i)
        data_visualization(df,df_clean, i)
