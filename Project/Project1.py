from sklearn.model_selection import GridSearchCV
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
import sklearn.metrics as met
import seaborn as sns
import time

#---------------------Functions------------------------------

def data_visualization(data, data_clean, coll):
    #print(df.describe())
    """ data[coll].plot()
    plt.title(coll)
    plt.show() """
    fig, axs = plt.subplots(1,2)
    fig.suptitle(coll)
    axs[0].plot(data[coll])
    axs[0].set_title('histogram without outliers')
    axs[1].hist(data[coll])
    axs[1].set_title( 'histogram with outliers')
    #axs[2].plot(data[coll])
    #axs[2].set_title('with outliers')
    #fig.title(coll)
    plt.show()

    return

def identify_outliers(data,  coll):

    upper_limit = data[coll].mean() + 5.4*data[coll].std()
    lower_limit = data[coll].mean() - 5.4*data[coll].std()
    out = data[coll][(data[coll] > upper_limit) | (data[coll] < lower_limit)]
    print(out)
    miss_values = data[data[coll].isnull()]
    return out, miss_values

def replacing_outliers_missvalues(data, outlier,miss_value, coll):
    data.iloc[outlier.index, data.columns == coll ] = data.iloc[outlier.index - 1, data.columns == coll ] 
    data.iloc[miss_value.index, data.columns == coll ] = data.iloc[miss_value.index - 1, data.columns == coll ]
    #data.iloc[miss_value.index, data.columns == coll ] = mean_miss
    #data.iloc[outlier.index, data.columns == coll ] = np.nan
    #data.interpolate(method = 'linear')
    return data

def split_data(x, y):

    Xtrain, Xtest, ytrain, ytest = train_test_split(x,y, test_size=0.15,shuffle=True, random_state=5 )
    return Xtrain, Xtest, ytrain, ytest

def normalize_data(xtrain, xtest):
    
    scaler = preprocessing.StandardScaler()
    scaler.fit(xtrain)
    x_n_train = scaler.transform(xtrain)
    x_n_test = scaler.transform(xtest)

    return x_n_test, x_n_train

def balance_data(x, y):

    oversample = SMOTE()
    X, Y = oversample.fit_resample(x, y)
    return X, Y

def hyperparameters_tunning(x_training, y_training):
    parameters={ 'hidden_layer_sizes':[(6,),(6,2), (10,), (14,) ], 'activation':[  'relu'], 'alpha':[  0.001, 0.0005, 0.0001], 'learning_rate_init':[ 0.003,0.002, 0.001]}
    scoring = {'accuracy':met.make_scorer(met.accuracy_score)}
    model = MLPClassifier(early_stopping=True)
    gs_cv = GridSearchCV(model,parameters,scoring=scoring, cv= 5,refit="accuracy")
    gs_cv.fit(x_training,y_training)
    best_params = gs_cv.best_params_
    print(best_params)
    
def prediction_multiclass(xtrain, ytrain, xtest, ytest):

    accuracy = {'model 1':[],'model 2':[],'model 3':[],'model 4':[]}
    precision = {'model 1':[],'model 2':[],'model 3':[],'model 4':[]}
    recall = {'model 1':[],'model 2':[],'model 3':[],'model 4':[]}
    Time = {'model 1':[],'model 2':[],'model 3':[],'model 4':[]}
    
    start_time1 = time.process_time()
    model1 = MLPClassifier(alpha=0.001, hidden_layer_sizes=(6,), learning_rate_init=0.02)
    model1.fit(xtrain, ytrain)
    y_pred1 = model1.predict(xtest)
    Time['model 1'].append(time.process_time() - start_time1)
    accuracy['model 1'].append(met.accuracy_score(ytest,y_pred1))
    precision['model 1'].append(met.precision_score(ytest, y_pred1,  average='macro'))
    recall['model 1'].append(met.recall_score(ytest,y_pred1,  average='macro'))

    start_time2 = time.process_time()
    model2 = MLPClassifier(alpha=0.0005, hidden_layer_sizes=(10,), learning_rate_init=0.004)
    model2.fit(xtrain, ytrain)
    y_pred2 = model2.predict(xtest)
    Time['model 2'].append(time.process_time() - start_time2)
    accuracy['model 2'].append(met.accuracy_score(ytest,y_pred2))
    precision['model 2'].append(met.precision_score(ytest, y_pred2,  average='macro'))
    recall['model 2'].append(met.recall_score(ytest,y_pred2,  average='macro'))

    start_time3 = time.process_time()
    model3 = MLPClassifier(alpha=0.0005, hidden_layer_sizes=(14,), learning_rate_init=0.01)
    model3.fit(xtrain, ytrain)
    y_pred3 = model3.predict(xtest)
    Time['model 3'].append(time.process_time() - start_time3)
    accuracy['model 3'].append(met.accuracy_score(ytest,y_pred3))
    precision['model 3'].append(met.precision_score(ytest, y_pred3,  average='macro'))
    recall['model 3'].append(met.recall_score(ytest,y_pred3,  average='macro'))

    start_time4 = time.process_time()
    model4 = MLPClassifier(alpha=0.0005, hidden_layer_sizes=(6,2), learning_rate_init=0.02)
    model4.fit(xtrain, ytrain)
    y_pred4 = model4.predict(xtest)
    Time['model 4'].append(time.process_time() - start_time4)
    accuracy['model 4'].append(met.accuracy_score(ytest,y_pred4))
    precision['model 4'].append(met.precision_score(ytest, y_pred4,  average='macro'))
    recall['model 4'].append(met.recall_score(ytest,y_pred4,  average='macro'))
    print('Multiclass Classification')
    print('accuracy: ', accuracy)
    print('precision: ', precision)
    print('recall: ', recall)
    print('Time: ', Time)
    return

def prediction_binary(xtrain, ytrain, xtest, ytest):
    accuracy = {'model 1':[],'model 2':[],'model 3':[],'model 4':[]}
    precision = {'model 1':[],'model 2':[],'model 3':[],'model 4':[]}
    recall = {'model 1':[],'model 2':[],'model 3':[],'model 4':[]}
    Time = {'model 1':[],'model 2':[],'model 3':[],'model 4':[]}
    
    start_time1 = time.process_time()
    model1 = MLPClassifier(alpha=0.0005, hidden_layer_sizes=(6,2), learning_rate_init=0.003)
    model1.fit(xtrain, ytrain)
    y_pred1 = model1.predict(xtest)
    Time['model 1'].append(time.process_time() - start_time1)
    accuracy['model 1'].append(met.accuracy_score(ytest,y_pred1))
    precision['model 1'].append(met.precision_score(ytest, y_pred1,  average='macro'))
    recall['model 1'].append(met.recall_score(ytest,y_pred1,  average='macro'))

    start_time2 = time.process_time()
    model2 = MLPClassifier(alpha=0.0005, hidden_layer_sizes=(10,), learning_rate_init=0.003)
    model2.fit(xtrain, ytrain)
    y_pred2 = model2.predict(xtest)
    Time['model 2'].append(time.process_time() - start_time2)
    accuracy['model 2'].append(met.accuracy_score(ytest,y_pred2))
    precision['model 2'].append(met.precision_score(ytest, y_pred2,  average='macro'))
    recall['model 2'].append(met.recall_score(ytest,y_pred2,  average='macro'))

    start_time3 = time.process_time()
    model3 = MLPClassifier(alpha=0.001, hidden_layer_sizes=(6,2), learning_rate_init=0.002)
    model3.fit(xtrain, ytrain)
    y_pred3 = model3.predict(xtest)
    Time['model 3'].append(time.process_time() - start_time3)
    accuracy['model 3'].append(met.accuracy_score(ytest,y_pred3))
    precision['model 3'].append(met.precision_score(ytest, y_pred3,  average='macro'))
    recall['model 3'].append(met.recall_score(ytest,y_pred3,  average='macro'))

    start_time4 = time.process_time()
    model4 = MLPClassifier(alpha=0.0005, hidden_layer_sizes=(14,), learning_rate_init=0.003)
    model4.fit(xtrain, ytrain)
    y_pred4 = model4.predict(xtest)
    Time['model 4'].append(time.process_time() - start_time4)
    accuracy['model 4'].append(met.accuracy_score(ytest,y_pred4))
    precision['model 4'].append(met.precision_score(ytest, y_pred4,  average='macro'))
    recall['model 4'].append(met.recall_score(ytest,y_pred4,  average='macro'))
    print('\nBinary Classification')
    print('accuracy: ', accuracy)
    print('precision: ', precision)
    print('recall: ', recall)
    print('Time: ', Time)
    return
#---------------------MAIN-----------------------------------

df = pd.read_csv('Proj1_Dataset.csv')
#df['Time'] = pd.to_datetime(df['Time'])
#df['Date'] = pd.to_datetime(df['Date'])

#print(df.info())
#print(df.describe())
#print(df.isnull().sum()) 
#print(df.describe())

df_clean = df.copy()
df_clean.drop(['Date', 'Time'] , axis=1, inplace=True)
for i in df_clean.columns:
    
    #if i != "Time" and i !="Date":  
    outliers, missing_values = identify_outliers(df_clean, i)
    df_clean = replacing_outliers_missvalues(df_clean, outliers, missing_values, i)
    #data_visualization(df,df_clean, i)

#df_clean.boxplot(column=['S1Temp', 'S2Temp', 'S3Temp' ])
#plt.show()

#plt.scatter(df_clean['S1Temp'][0:100], df_clean['Time'][0:100], marker='^')
#plt.scatter(df_clean['S2Temp'][0:100], df_clean['Time'][0:100], marker='o')
#plt.scatter(df_clean['S3Temp'][0:100], df_clean['Time'][0:100], marker='x')
#plt.show()

#Correlations

plt.figure(figsize=(12,10), dpi= 80)
sns.heatmap(df_clean.corr(), xticklabels=df_clean.corr().columns, yticklabels=df_clean.corr().columns, cmap='RdYlGn', center=0, annot=True)
plt.title('Correlogram heatmap', fontsize=22)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

# X and Y data 

y = df_clean['Persons'].copy()
x = df_clean.drop(columns=['Persons'], axis= 1 , inplace=False)

#SPLIT DATA

x_train, x_test, y_train, y_test = split_data(x, y)

#NORMALIZE DATA

x_norm_test, x_norm_train = normalize_data(x_train, x_test)

#BALANCE TRAINING DATA

#counter = Counter(y_train)
#print(counter)

X_train, Y_train = balance_data(x_norm_train, y_train)
y_train_binary = np.where(Y_train>2, 1, 0 )
y_test_binary = np.where(y_test>2, 1, 0 )


""" plt.figure(figsize=(12,10), dpi= 80)
sns.heatmap(X_train.corr(), xticklabels=X_train.corr().columns, yticklabels=X_train.corr().columns, cmap='RdYlGn', center=0, annot=True)
plt.title('Correlogram heatmap', fontsize=22)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show() """


#Hyperparameters tunning

#hyperparameters_tunning(X_train,Y_train )
#hyperparameters_tunning(X_train, y_train_binary )

#PREDICTION RESULTS

prediction_multiclass(X_train, Y_train, x_norm_test,y_test )
prediction_binary(X_train, y_train_binary,x_norm_test,y_test_binary  )
