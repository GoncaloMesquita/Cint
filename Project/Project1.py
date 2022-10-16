from sklearn.model_selection import GridSearchCV
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
import sklearn.metrics as met
import seaborn as sns
import time
from imblearn.pipeline import Pipeline as imbpipeline

#---------------------Functions------------------------------

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

def split_data(x, y):

    Xtrain, Xtest, ytrain, ytest = train_test_split(x,y, test_size=0.15,random_state=42)

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

def hyperparameters_tunning(x_training, y_training, unbalanced, bin):
     
    if bin == 1:
        y_training = np.where(y_training>2, 1, 0 )
        scoring = {'score': met.make_scorer(met.recall_score)}
    else:
        scoring = {'score': met.make_scorer(met.accuracy_score)}

    if unbalanced == 1:
        parameters={ 'classifier__hidden_layer_sizes':[(6,),(6,2), (10,), (14,)],  'classifier__alpha':[  0.001, 0.0005], 'classifier__learning_rate_init':[ 0.003,0.002, 0.001]}
        pipeline = imbpipeline(steps = [['smote', SMOTE(random_state=11)],['classifier', MLPClassifier()]])
        gs_cv = GridSearchCV(pipeline , parameters, scoring=scoring, cv= 5,refit="score")
        gs_cv.fit(x_training,y_training)
        print(gs_cv.best_params_)
    else :
        parameters={ 'hidden_layer_sizes':[(6,),(6,2), (10,), (14,)],  'alpha':[  0.001, 0.0005], 'learning_rate_init':[ 0.003,0.002, 0.001]}
        model = MLPClassifier()
        gs_cv1 = GridSearchCV(model , parameters, scoring=scoring, cv= 5,refit="score")
        gs_cv1.fit(x_training,y_training)
        print(gs_cv1.best_params_)
    
    return
    
def prediction_multiclass(xtrain, ytrain, xtest, ytest, unbalanced):

    if unbalanced == 1:
        xtrain, ytrain = balance_data(xtrain, ytrain)
        
    accuracy = {'model 1':[],'model 2':[],'model 3':[],'model 4':[]}
    precision = {'model 1':[],'model 2':[],'model 3':[],'model 4':[]}
    recall = {'model 1':[],'model 2':[],'model 3':[],'model 4':[]}
    Time = {'model 1':[],'model 2':[],'model 3':[],'model 4':[]}
    C_matrix = {'model 1':[],'model 2':[],'model 3':[],'model 4':[]}

    start_time1 = time.process_time()
    model1 = MLPClassifier(alpha=0.001, hidden_layer_sizes=(6,), learning_rate_init=0.02)
    model1.fit(xtrain, ytrain)
    y_pred1 = model1.predict(xtest)
    Time['model 1'].append(time.process_time() - start_time1)
    accuracy['model 1'].append(met.accuracy_score(ytest,y_pred1))
    precision['model 1'].append(met.precision_score(ytest, y_pred1,  average='macro'))
    recall['model 1'].append(met.recall_score(ytest,y_pred1,  average='macro'))
    C_matrix['model 1'].append(met.confusion_matrix(ytest,y_pred1))

    start_time2 = time.process_time()
    model2 = MLPClassifier(alpha=0.0005, hidden_layer_sizes=(10,), learning_rate_init=0.003)
    model2.fit(xtrain, ytrain)
    y_pred2 = model2.predict(xtest)
    Time['model 2'].append(time.process_time() - start_time2)
    accuracy['model 2'].append(met.accuracy_score(ytest,y_pred2))
    precision['model 2'].append(met.precision_score(ytest, y_pred2,  average='macro'))
    recall['model 2'].append(met.recall_score(ytest,y_pred2,  average='macro'))
    C_matrix['model 2'].append(met.confusion_matrix(ytest,y_pred2))

    start_time3 = time.process_time()
    model3 = MLPClassifier(alpha=0.0001, hidden_layer_sizes=(14,),activation = 'relu', learning_rate_init=0.002)
    model3.fit(xtrain, ytrain)
    y_pred3 = model3.predict(xtest)
    Time['model 3'].append(time.process_time() - start_time3)
    accuracy['model 3'].append(met.accuracy_score(ytest,y_pred3))
    precision['model 3'].append(met.precision_score(ytest, y_pred3,  average='macro'))
    recall['model 3'].append(met.recall_score(ytest,y_pred3,  average='macro'))
    C_matrix['model 3'].append(met.confusion_matrix(ytest,y_pred3))

    start_time4 = time.process_time()
    model4 = MLPClassifier(alpha=0.0005, hidden_layer_sizes=(6,6),activation='relu', learning_rate_init=0.003)
    model4.fit(xtrain, ytrain)
    y_pred4 = model4.predict(xtest)
    Time['model 4'].append(time.process_time() - start_time4)
    accuracy['model 4'].append(met.accuracy_score(ytest,y_pred4))
    precision['model 4'].append(met.precision_score(ytest, y_pred4,  average='macro'))
    recall['model 4'].append(met.recall_score(ytest,y_pred4,  average='macro'))
    C_matrix['model 4'].append(met.confusion_matrix(ytest,y_pred4))

    print('Multiclass Classification')
    print('accuracy: ', accuracy)
    print('precision: ', precision)
    print('recall: ', recall)
    print('Confusion Matrix:', C_matrix)
    print('Time: ', Time)
    return model4

def prediction_binary(xtrain, ytrain, xtest, ytest, unbalanced):
    
    if unbalanced == 1:
        xtrain, ytrain = balance_data(xtrain, ytrain)

    ytrain = np.where(ytrain>2, 1, 0 )
    ytest = np.where(ytest>2, 1, 0 )
    accuracy = {'model 1':[],'model 2':[],'model 3':[],'model 4':[]}
    precision = {'model 1':[],'model 2':[],'model 3':[],'model 4':[]}
    recall = {'model 1':[],'model 2':[],'model 3':[],'model 4':[]}
    Time = {'model 1':[],'model 2':[],'model 3':[],'model 4':[]}
    C_matrix = {'model 1':[],'model 2':[],'model 3':[],'model 4':[]}
    
    start_time1 = time.process_time()
    model1 = MLPClassifier(alpha=0.0005, hidden_layer_sizes=(100,100), learning_rate_init=0.001)
    model1.fit(xtrain, ytrain)
    y_pred1 = model1.predict(xtest)
    Time['model 1'].append(time.process_time() - start_time1)
    accuracy['model 1'].append(met.accuracy_score(ytest,y_pred1))
    precision['model 1'].append(met.precision_score(ytest, y_pred1,  average=None))
    recall['model 1'].append(met.recall_score(ytest,y_pred1,  average=None))
    C_matrix['model 1'].append(met.confusion_matrix(ytest,y_pred1))

    start_time2 = time.process_time()
    model2 = MLPClassifier(alpha=0.0005, hidden_layer_sizes=(10,), learning_rate_init=0.003)
    model2.fit(xtrain, ytrain)
    y_pred2 = model2.predict(xtest)
    Time['model 2'].append(time.process_time() - start_time2)
    accuracy['model 2'].append(met.accuracy_score(ytest,y_pred2))
    precision['model 2'].append(met.precision_score(ytest, y_pred2,  average=None))
    recall['model 2'].append(met.recall_score(ytest,y_pred2,  average=None))
    C_matrix['model 2'].append(met.confusion_matrix(ytest,y_pred2))

    start_time3 = time.process_time()
    model3 = MLPClassifier(alpha=0.001, hidden_layer_sizes=(6,), learning_rate_init=0.002)
    model3.fit(xtrain, ytrain)
    y_pred3 = model3.predict(xtest)
    Time['model 3'].append(time.process_time() - start_time3)
    accuracy['model 3'].append(met.accuracy_score(ytest,y_pred3))
    precision['model 3'].append(met.precision_score(ytest, y_pred3,  average=None))
    recall['model 3'].append(met.recall_score(ytest,y_pred3,  average=None))
    C_matrix['model 3'].append(met.confusion_matrix(ytest,y_pred3))

    start_time4 = time.process_time()
    model4 = MLPClassifier(alpha=0.0005, hidden_layer_sizes=(14,), learning_rate_init=0.003)
    model4.fit(xtrain, ytrain)
    y_pred4 = model4.predict(xtest)
    Time['model 4'].append(time.process_time() - start_time4)
    accuracy['model 4'].append(met.accuracy_score(ytest,y_pred4))
    precision['model 4'].append(met.precision_score(ytest, y_pred4,  average=None))
    recall['model 4'].append(met.recall_score(ytest,y_pred4,  average=None))
    C_matrix['model 4'].append(met.confusion_matrix(ytest,y_pred4))

    print('\nBinary Classification')
    print('accuracy: ', accuracy)
    print('precision: ', precision)
    print('recall: ', recall)
    print('Confusion Matrix:', C_matrix)
    print('Time: ', Time)

    return model3

######################################  MAIN   #################################################

df = pd.read_csv('Proj1_Dataset.csv')

#################################### Outliers, Missing Data and Feature Selection ###############

df_clean = df.copy()
df_clean.drop(['Date', 'Time', 'CO2', 'S3Temp'] , axis=1, inplace=True)

for i in df_clean.columns:
     
    outliers, missing_values = identify_outliers(df_clean, i)
    df_clean = replacing_outliers_missvalues(df_clean, outliers, missing_values, i)

####################################### X and Y data ##########################################

y = df_clean['Persons'].copy()
x = df_clean.drop(columns=['Persons'], axis= 1 , inplace=False)

####################################### SPLIT DATA   ###########################################

x_train, x_test, y_train, y_test = split_data(x, y)

#######################################  NORMALIZE DATA   #####################################

x_norm_test, x_norm_train = normalize_data(x_train, x_test)

#######################################  Hyperparameters tunning #################################

#hyperparameters_tunning(x_norm_train,y_train, 0, 1)

#######################################  PREDICTION RESULTS  ######################################

#model_multi= prediction_multiclass(x_norm_train, y_train, x_norm_test, y_test, 0)
model_multi= prediction_multiclass(x_norm_train, y_train, x_norm_test, y_test, 1)
#model_binary = prediction_binary(x_norm_train, y_train, x_norm_test, y_test, 0)

from joblib import dump, load
#dump(model_multi, 'Model_Multi_class.joblib')
#dump(model_binary, 'Model_Binary.joblib')
