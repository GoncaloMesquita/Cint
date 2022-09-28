# -*- coding: utf-8 -*-
"""
Created on  September 27 2019

@author: Rui Neves

Lab 2 Computacional Inteligence
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

df = pd.read_csv('AAPL_yah.csv')

df1 = df.sort_values(by=['Close'])

df1.to_csv('ChangedData.csv')
#open the file to see what happened
#notice that the index was also written

#now control how you write to the file
df1.to_csv('ChangedData1.csv', decimal=',', sep=';', index=False)


start_time1 = time.process_time()
start_time2 = time.perf_counter()

df['newCol'] = df['Volume'].cumsum()

print ("time used in cumsum -->", time.process_time() - start_time1, "seconds")
print ("time used in cumsum -->", time.perf_counter() - start_time2, "seconds")

a = np.arange(10)
print(a)
#a = array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
b = np.where(a < 5, a, 10*a)
print(b)
#array([ 0,  1,  2,  3,  4, 50, 60, 70, 80, 90])

df['signal']= np.where(df['Close'] > df['Open'] , 1.0, 0.0)  



# First example, my first plot
plt.plot([1, 2, 3, 4])
plt.ylabel('some numbers')
plt.show()

# second example, lets control the axis
plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'ro')
plt.axis([0, 6, 0, 20])
plt.show()

# third example, lets have two plots in one figure
def f(t):
	return np.exp(-t) * np.cos(2*np.pi*t)


t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02) 


plt.figure()
plt.subplot(211)
plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')

plt.subplot(212)
plt.plot(t2, np.cos(2*np.pi*t2), 'r--')
plt.show() 



end=0

""" Exercise 4 """
data = pd.read_csv('DCOILBRENTEUv2.csv')

norm = (data['DCOILBRENTEU']-data['DCOILBRENTEU'].min())/(data['DCOILBRENTEU'].max()-data['DCOILBRENTEU'].min())
zscore = (data['DCOILBRENTEU']-data['DCOILBRENTEU'].mean())/(data['DCOILBRENTEU'].std())
plt.figure()

plt.plot(norm) 
plt.plot(zscore)
plt.show()

sma = norm.rolling(50).mean()
plt.figure()
plt.plot(sma)
plt.show()

