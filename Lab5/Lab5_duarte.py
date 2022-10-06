from statistics import mean
import sklearn.metrics as met
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier

# Haberman data set

hab_dat = pd.read_table('haberman.data',sep=',',header=None)
hab_dat.columns = ['Age', 'Year of operation', 'Nodes detected', 'Survival status']

# Iris data set
iris_dat = pd.read_table('iris.data',sep=',',header=None)
iris_dat.columns = ['sepal length','sepal width','petal length','petal width','class']


# Separate training set from test set
X_Hab_train, X_Hab_test, y_Hab_train, y_Hab_test = train_test_split(hab_dat.iloc[:,:3].values,hab_dat.iloc[:,3].values, test_size=0.2,random_state=20)

X_iris_train, X_iris_test, y_iris_train, y_iris_test = train_test_split(iris_dat.iloc[:,:4].values,iris_dat.iloc[:,4].values, test_size=0.2,random_state=20)

def data_normalizer(x_train, x_test):
    scaler = preprocessing.StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, x_test

def hyper_param_tunning(x_train, y_train):
    alpha = 0.01
    alphas = [alpha]
    learn_rate = 0.01
    l_rates =[learn_rate]
    for i in range(10):
        alpha = alpha/1.35
        learn_rate = learn_rate/1.4
        alphas.append(alpha)
        l_rates.append(learn_rate)

    param_grid = {'hidden_layer_sizes':[(3,),(4,),(5,),(6,),(3,3),(4,4)], 'activation':['identity', 'logistic', 'tanh', 'relu'], 'alpha':alphas, 'learning_rate_init':l_rates}
    scoring = {'accuracy':met.make_scorer(met.accuracy_score)}
    mlp = MLPClassifier(solver='lbfgs', max_iter=10000, max_fun=150000)
    gs_cv = GridSearchCV(mlp,param_grid,scoring=scoring, cv= 5,refit="accuracy")
    gs_cv.fit(x_train,y_train)
    best_params = gs_cv.best_params_

    return best_params

# x_train , x_test = data_normalizer(X_Hab_train, X_Hab_test)
# print(hyper_param_tunning(x_train,y_Hab_train))

def scores(x_train, x_test, y_train, y_test,data_set_name):
    x_train, x_test =  data_normalizer(x_train, x_test)
    
    if data_set_name == 'iris':
        # hyperparams = {'activation': 'logistic', 'alpha': 0.0012236680494047743, 'hidden_layer_sizes': (3,), 'learning_rate_init': 0.0071428571428571435}
        # hyperparams = {'activation': 'logistic', 'alpha': 0.0022301350200402015, 'hidden_layer_sizes': (3,), 'learning_rate_init': 0.0026030820491461898}
        hyperparams = {'activation': 'logistic', 'alpha': 0.0022301350200402015, 'hidden_layer_sizes': (3, 3), 'learning_rate_init': 0.0009486450616421977}
        average = 'macro'
    elif data_set_name == 'haberman':
        hyperparams = {'activation': 'relu', 'alpha': 0.0022301350200402015, 'hidden_layer_sizes': (3,), 'learning_rate_init': 0.0004840025824705091}
        average = 'binary'
        
    mlp = MLPClassifier(activation=hyperparams['activation'],hidden_layer_sizes=hyperparams['hidden_layer_sizes'],alpha=hyperparams['alpha'], learning_rate_init=hyperparams['learning_rate_init'], solver='lbfgs', max_iter=10000)
    mlp.fit(x_train,y_train)
    y_pred = mlp.predict(x_test)
    
    score = {'accuracy':met.accuracy_score(y_test, y_pred),'precision':met.precision_score(y_test, y_pred, average=average),'recall':met.recall_score(y_test, y_pred,average=average)}
    return score

# print(scores(X_iris_train,X_iris_test, y_iris_train, y_iris_test, 'iris'))
print(scores(X_Hab_train,X_Hab_test, y_Hab_train, y_Hab_test, 'haberman'))