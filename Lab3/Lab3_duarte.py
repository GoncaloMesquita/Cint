from cProfile import label
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np


""" 1 task """
df_eur = pd.read_csv('EURUSD_Daily_Ask_2018.12.31_2019.10.05v2.csv', sep=';',decimal=',')
df_eur['Time (UTC)'] = pd.to_datetime(df_eur['Time (UTC)'])

def plot(df):
    plt.figure()
    plt.plot(df['Time (UTC)'],df['Open'], label='Open')
    plt.plot(df['Time (UTC)'],df['Close'], label='Close')
    plt.plot(df['Time (UTC)'],df['High'], label='High')
    plt.plot(df['Time (UTC)'],df['Low'], label='Low')
    plt.legend()
    plt.show()    

    plt.figure()
    plt.plot(df['Time (UTC)'],df['Volume '], label='Volume ')
    plt.legend()
    plt.show() 

def outlier_removal_1(df:pd.DataFrame):
    df = df.drop(df.index[df['Time (UTC)'] == '2019-05-29 21:00:00'])
    plot(df)

# outlier_removal_1(df_eur)

def outlier_removal_2(df:pd.DataFrame):
    # print(df.loc[:, df.columns != 'Time (UTC)'].std())
    # print(df.columns.to_list())
    for col in df.columns:
        if col != 'Time (UTC)':
            # print(df.index[(df[col] >= df[col].mean() + 2*df[col].std()) | (df[col] <= df[col].mean() - 2*df[col].std())])
            df = df.drop(df.index[(df[col] >= df[col].mean() + 3*df[col].std()) | (df[col] <= df[col].mean() - 3*df[col].std())])
    plot(df)      


# outlier_removal_2(df_eur)

def outlier_removal_3(df:pd.DataFrame):
    for col in df.columns:
        if col != 'Time (UTC)' :
            # print(df.index[(df[col] >= df[col].mean() + 2*df[col].std()) | (df[col] <= df[col].mean() - 2*df[col].std())])
            index = df.index[(df[col] >= df[col].mean() + 3*df[col].std()) | (df[col] <= df[col].mean() - 3*df[col].std())]
            # print(df.iloc[index])
            df.iloc[index, df.columns == col] = df.iloc[np.array(index)-1, df.columns == col]
            # print(df.iloc[index])
    plot(df)       

# outlier_removal_3(df_eur)

def outlier_removal_4(df:pd.DataFrame):
    for col in df.columns:
        if col != 'Time (UTC)' :
            
            index = df.index[(df[col] >= df[col].mean() + 3*df[col].std()) | (df[col] <= df[col].mean() - 3*df[col].std())]
            # print(df.iloc[index])
            # print(pd.Series([df.iloc[np.array(index)-1, df.columns == col],df.iloc[np.array(index)+1, df.columns == col]]).interpolate())
            df.iloc[index, df.columns == col] = (np.array(df.iloc[np.array(index)-1, df.columns == col]) + np.array(df.iloc[np.array(index)+1, df.columns == col]))/2
            # print(df.iloc[index])
    plot(df)   
# outlier_removal_4(df_eur)

""" 2 task """

df_2 = pd.read_csv('DCOILBRENTEUv2.csv', sep=',',decimal='.')


# plt.hist(np.diff(df_2['DCOILBRENTEU']), bins=35)
# plt.show()

""" 3 task """

df_3 = pd.read_csv('DCOILWTICOv2.csv', sep=',',decimal='.')


plt.scatter(df_2['DATE'],df_2['DCOILBRENTEU'])
plt.scatter(df_3['DATE'],df_3['DCOILWTICO'])
plt.show()