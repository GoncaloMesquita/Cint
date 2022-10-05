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

# Number of k folds for cross validation

K = 10

# Separate training set from test set
X_Hab_train, X_Hab_test, y_Hab_train, y_Hab_test = train_test_split(hab_dat.iloc[:,:3].values,hab_dat.iloc[:,3].values, test_size=0.1,random_state=20)

X_iris_train, X_iris_test, y_iris_train, y_iris_test = train_test_split(iris_dat.iloc[:,:4].values,iris_dat.iloc[:,4].values, test_size=0.1,random_state=20)

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

    param_grid = {'hidden_layer_sizes':[(3,),(4,),(5,),(6,),(3,3),(4,4),(3,4),(4,5)], 'activation':['identity', 'logistic', 'tanh', 'relu'], 'alpha':alphas, 'learning_rate_init':l_rates}
    scoring = {'accuracy':met.accuracy_score,'precision':met.precision_score,'recall':met.recall_score}
    mlp = MLPClassifier(solver='lbfgs')
    best_params = GridSearchCV(mlp,param_grid,scoring=scoring, cv= 5).best_params_
    return best_params

