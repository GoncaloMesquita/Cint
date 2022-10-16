
from joblib import  load
import pandas as pd
import sys
from sklearn import preprocessing
import time
import sklearn.metrics as met

import numpy as np

##################################### Functions #######################

def identify_outliers(data,  coll):

    upper_limit = data[coll].mean() + 5.4*data[coll].std()
    lower_limit = data[coll].mean() - 5.4*data[coll].std()
    out = data[coll][(data[coll] > upper_limit) | (data[coll] < lower_limit)]
    miss_values = data[data[coll].isnull()]
    
    return out, miss_values

def replacing_outliers_missvalues(data, outlier,miss_value, coll):

    for index in outlier.index:
        data.iloc[index, data.columns == coll ] = data.iloc[index - 1, data.columns == coll] 
    for index in miss_value.index:
        data.iloc[index, data.columns == coll ] = data.iloc[index - 1, data.columns == coll]

    return data

def normalize_data(X):
    
    scaler = preprocessing.StandardScaler()
    scaler.fit(X)
    X_data = scaler.transform(X)

    return X_data

def predict_multi(X, Y, model):

    y_pred = model.predict(X)
    precision_macro = (met.precision_score(Y, y_pred,  average='macro'))
    recall_macro = (met.recall_score(Y, y_pred,  average='macro'))
    precision = (met.precision_score(Y, y_pred,  average=None))
    recall = met.recall_score(Y,y_pred,  average=None)
    C_matrix = met.confusion_matrix(Y,y_pred)
    f1_score_macro = met.f1_score(Y,y_pred, average='macro')
    print('\nMulti Classification')
    print('Confusion Matrix:\n ', C_matrix)
    print('Precision: ', precision)
    print('Recall: ', recall)
    print('Precision macro: ', precision_macro)
    print('Recall macro: ', recall_macro)
    print('F1 macro: ', f1_score_macro)

    return

def predict_binary(X, Y, model):

    y_binary = np.where(Y > 2, 1,0)
    y_pred = model.predict(X)
    precision_macro = (met.precision_score(y_binary, y_pred,  average='macro'))
    recall_macro = (met.recall_score(y_binary, y_pred,  average='macro'))
    precision = (met.precision_score(y_binary, y_pred,  average=None))
    recall = met.recall_score(y_binary,y_pred,  average=None)
    C_matrix = met.confusion_matrix(y_binary,y_pred)
    f1_score_macro = met.f1_score(y_binary,y_pred, average='macro')
    print('\nMulti Binary')
    print('Confusion Matrix: \n', C_matrix)
    print('Precision: ', precision)
    print('Recall: ', recall)
    print('Precision macro: ', precision_macro)
    print('Recall macro: ', recall_macro)
    print('F1 macro: ', f1_score_macro)

    return

##################################### MAIN ############################

if len(sys.argv) < 2:
    print(" ----- Error: arguments should follow the following setup ----- ")
    print("(python) TestMe.py xxx.csv")
    exit(0)
elif sys.argv[1][int(len(sys.argv[1])-4):int(len(sys.argv[1]))] != '.csv':
        print(" ----- Error: arguments should follow the following setup ----- ")
        print("(python) TestMe.py xxx.csv")
else:
    df = pd.read_csv(sys.argv[1])

model_multi = load('Model_Multi_class.joblib') 
model_binary = load('Model_Binary.joblib')

#################################### Outliers, Missing Data and Feature Selection ###############

df_clean = df.copy()
df_clean.drop(['Date', 'Time', 'CO2', 'S3Temp'] , axis=1, inplace=True)

for i in df_clean.columns:
     
    outliers, missing_values = identify_outliers(df_clean, i)
    df_clean = replacing_outliers_missvalues(df_clean, outliers, missing_values, i)

####################################### X and Y data ##########################################

y = df_clean['Persons'].copy()
x = df_clean.drop(columns=['Persons'], axis= 1 , inplace=False)

#######################################  NORMALIZE DATA   #####################################

x_norm = normalize_data(x)

#######################################  PREDICT RESULTS  ######################################

predict_multi(x_norm, y,model_multi)
predict_binary(x_norm, y, model_binary)