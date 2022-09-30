from statistics import mean
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import sklearn.metrics as met
from sklearn.model_selection import cross_val_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# Haberman

hab_dat = pd.read_table('haberman.data',sep=',',header=None)
hab_dat.columns = ['Age', 'Year of operation', 'Nodes detected', 'Survival status']

# Iris
iris_dat = pd.read_table('iris.data',sep=',',header=None)
iris_dat.columns = ['sepal length','sepal width','petal length','petal width','class']



#Plots
sns.set_theme(style="darkgrid")
sns.relplot(iris_dat,x='sepal length',y='sepal width',hue='class',style='class', size='petal width')

sns.relplot(hab_dat,x='Age',y='Nodes detected',hue='Survival status',style='Survival status',size='Nodes detected')
# plt.show()




accuracy = {'bayes':[],'LinearSVC':[],'svc':[],'k_neighbors':[]}
precision = {'bayes':[],'LinearSVC':[],'svc':[],'k_neighbors':[]}
recall = {'bayes':[],'LinearSVC':[],'svc':[],'k_neighbors':[]}
for i in range(1):
    X_train, X_test, y_train, y_test = train_test_split(hab_dat.iloc[:,:3].values,hab_dat.iloc[:,3].values, test_size=0.1,random_state=42 )
    # train_hab = pd.DataFrame({'Age':[X_train[:,0],y_train[:,0]], 'Year of operation', 'Nodes detected', 'Survival status'})

    # X_train, X_validation, y_train, y_validation = train_test_split(hab_dat.iloc[:,:3].values,hab_dat.iloc[:,3].values, test_size=.2,random_state=30)
    

    # Naive Bayes
    bayes_clf = GaussianNB()
    bayes_clf.fit(X_train, y_train)
    y_bayes = bayes_clf.predict(X_test)
    accuracy['bayes'].append(met.accuracy_score(y_test,y_bayes))
    precision['bayes'].append(met.precision_score(y_test,y_bayes))
    recall['bayes'].append(met.recall_score(y_test,y_bayes))
    # conf_mat['bayes'] = met.confusion_matrix(y_test,y_bayes)

    # LinearSVC
    LinearSVC_clf = LinearSVC()
    LinearSVC_clf.fit(X_train, y_train)
    y_LinearSVC = LinearSVC_clf.predict(X_test)
    accuracy['LinearSVC'].append(met.accuracy_score(y_test,y_LinearSVC))
    precision['LinearSVC'].append(met.precision_score(y_test,y_LinearSVC))
    recall['LinearSVC'].append(met.recall_score(y_test,y_LinearSVC))
    # conf_mat = met.confusion_matrix(y_test,y_LinearSVC)

    # SVC
    svc_clf = svm.SVC()
    svc_clf.fit(X_train, y_train)
    y_svc = svc_clf.predict(X_test)
    accuracy['svc'].append(met.accuracy_score(y_test,y_svc))
    precision['svc'].append(met.precision_score(y_test,y_svc))
    recall['svc'].append(met.recall_score(y_test,y_svc))
    # conf_mat = met.confusion_matrix(y_test,y_svc)

    # K-Neighbors
    k_neighbors_clf = KNeighborsClassifier()
    k_neighbors_clf.fit(X_train, y_train)
    y_k_neighbors = k_neighbors_clf.predict(X_test)
    accuracy['k_neighbors'].append(met.accuracy_score(y_test,y_k_neighbors))
    precision['k_neighbors'].append(met.precision_score(y_test,y_k_neighbors))
    recall['k_neighbors'].append(met.recall_score(y_test,y_k_neighbors))
    # conf_mat = met.confusion_matrix(y_test,y_k_neighbors)
    


# def cross


# print(conf_mat)
print('accuracy: ', accuracy)
print('precision: ', precision)
print('recall: ', recall)
