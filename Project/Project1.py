import pandas as pd
import matplotlib.pyplot as plt
import datetime


#---------------------Functions------------------------------

def data_visualization(data, coll):
    
    plt.plot(data['Time'], data[coll])
    plt.show()
#---------------------MAIN-----------------------------------

df = pd.read_csv('Proj1_Dataset.csv')

print(df.describe())

df['Time'] = pd.to_datetime(df['Time'])

for i in df.columns:
    
    if i != "Time" and i !="Date":
        print(i)
        data_visualization(df, i) 
