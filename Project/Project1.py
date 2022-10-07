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
from sklearn.model_selection import GridSearchCV

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

def identify_outliers(data,  coll):

    upper_limit = data[coll].mean() + 5.4*data[coll].std()
    lower_limit = data[coll].mean() - 5.4*data[coll].std()
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

def split_data(x, y):

    Xtrain, Xtest, ytrain, ytest = train_test_split(x,y, test_size=0.10,shuffle=True )
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

def cross_validation(xtraining, ytraining):
    start_time1 = time.process_time()
    clf = MLPClassifier( hidden_layer_sizes=(6, ),max_iter = 300, activation='relu', random_state=1, early_stopping=True)
    results = cross_validate(clf,X=xtraining,y=ytraining,cv=5, return_train_score=True,scoring='recall_macro'  ,return_estimator=True)
    print ("Time of crossvalidation ", time.process_time() - start_time1, "seconds")
    print('Cross Validation results: ', np.mean(results['test_score']),'\nCross Validation train results: ', np.mean(results['train_score']))
    return results

def hyperparameters_tunning(x_training, y_training):
    parameters={ 'hidden_layer_sizes':[(3,),(6,),(6,2),(10,), (13,) ], 'activation':[ 'logistic',  'relu'], 'alpha':[  0.001, 0.0005], 'learning_rate_init':[0.004, 0.003,0.002, 0.001]}
    scoring = {'accuracy':met.make_scorer(met.accuracy_score)}
    model = MLPClassifier(early_stopping=True)
    gs_cv = GridSearchCV(model,parameters,scoring=scoring, cv= 5,refit="accuracy")
    gs_cv.fit(x_training,y_training)
    best_params = gs_cv.best_params_
    print(best_params)

def prediction(modelos, X_test, Y_test):

    y_pred = modelos['estimator'][4].predict(X_test)
    print('Test Accuracy: ' ,met.accuracy_score(Y_test,y_pred))
    print('Test Precision of each class: ',met.precision_score(Y_test, y_pred,  average=None))
    print('Test Recall: ', met.recall_score(y_test,y_pred,  average='macro'))
    print('Confusion Matrix' ,met.confusion_matrix(Y_test,y_pred))

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

#CROSS VALIDATION 

#models = cross_validation(X_train, Y_train)

#Hyperparameters tunning

hyperparameters_tunning(X_train,Y_train )

#PREDICTION RESULTS

#prediction(models,x_norm_test, y_test)

