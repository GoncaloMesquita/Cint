from typing import Counter
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np
from matplotlib import pyplot
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
import sklearn.metrics as met

#---------------------Functions------------------------------

def data_visualization(data, data_clean, coll):
    #print(df.describe())
    """ data[coll].plot()
    plt.title(coll)
    plt.show() """
    fig, axs = plt.subplots(2)
    axs[0].plot(data[coll])
    axs[1].plot(data_clean[coll])
    #fig.title(coll)
    plt.show()

    return

def identify_outliers1(data,  coll):

    upper_limit = data[coll].mean() + 6*data[coll].std()
    lower_limit = data[coll].mean() - 6*data[coll].std()
    out = data[coll][(data[coll] > upper_limit) | (data[coll] < lower_limit)]
    miss_values = data[data[coll].isnull()]
    return out, miss_values

def replacing_outliers_missvalues(data, outlier,miss_value, coll):
    data.iloc[outlier.index, data.columns == coll ] = data.iloc[outlier.index - 1, data.columns == coll ] 
    data.iloc[miss_value.index, data.columns == coll ] = data.iloc[miss_value.index - 1, data.columns == coll ]
    #data.iloc[miss_value.index, data.columns == coll ] = mean_miss
    #data.iloc[outlier.index, data.columns == coll ] = np.nan
    #data.interpolate(method = 'linear')
    return data

def normalize_data(x_data):

    x_norm = preprocessing.normalize(x_data)
    return x_norm

def split_data(x, y):

    Xtrain, Xtest, ytrain, ytest = train_test_split(x,y, test_size=0.10,shuffle=True )
    return Xtrain, Xtest, ytrain, ytest

def balance_data(x, y):

    oversample = SMOTE()
    X, Y = oversample.fit_resample(x, y)
    return X, Y

#---------------------MAIN-----------------------------------

df = pd.read_csv('Proj1_Dataset.csv')
#df['Time'] = pd.to_datetime(df['Time'])
#df['Date'] = pd.to_datetime(df['Date'])

#print(df.info())
#print(df.describe())
#print(df.isnull().sum()) 

#print(df.describe())
df_clean = df.copy()
df_clean.drop(['Date', 'Time'], axis=1, inplace=True)
for i in df_clean.columns:
    
    #if i != "Time" and i !="Date":  
    outliers, missing_values = identify_outliers1(df_clean, i)
    df_clean = replacing_outliers_missvalues(df_clean, outliers, missing_values, i)
        #data_visualization(df,df_clean, i)

df_clean.boxplot(column=['S1Temp', 'S2Temp', 'S3Temp' ])
plt.show()

#plt.scatter(df_clean['S1Temp'][0:100], df_clean['Time'][0:100], marker='^')
#plt.scatter(df_clean['S2Temp'][0:100], df_clean['Time'][0:100], marker='o')
#plt.scatter(df_clean['S3Temp'][0:100], df_clean['Time'][0:100], marker='x')
#plt.show()

y = df_clean['Persons'].copy()
x = df_clean.drop(columns=['Persons'], axis= 1 , inplace=False)

#Nomalize data

X_norm = normalize_data(x)

#Slipt the data 

x_train, x_test, y_train, y_test = split_data(X_norm, y)

#Balance the training data
#counter = Counter(y_data)
#print(counter)
#X_train, Y_train = balance_data(x_train, y_train)
clf = MLPClassifier(  hidden_layer_sizes=(6, ),max_iter = 1000, activation='relu', random_state=1)
#_scoring = {'accuracy' : make_scorer(accuracy_score), 'precision' : make_scorer(precision_score),'recall' : make_scorer(recall_score), 'f1_score' : make_scorer(f1_score)}
results = cross_validate(clf,X=x_train,y=y_train,cv=5, return_train_score=True,scoring='recall_macro'  ,return_estimator=True)
best = 0
print(results)
for i in range ( 0, len(results['train_score'])):
    if best < results['train_score'][i]:
        index_best = i 
        best = results['train_score'][i]
y_pred = results['estimator'][index_best].predict(x_test)
print(met.accuracy_score(y_test,y_pred))
print(met.precision_score(y_test, y_pred,  average=None))
print(met.recall_score(y_test,y_pred,  average='macro'))
