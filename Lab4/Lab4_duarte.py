from statistics import mean
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import sklearn.metrics as met
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
# sns.set_theme(style="darkgrid")
# sns.relplot(iris_dat,x='sepal length',y='sepal width',hue='class',style='class', size='petal width')

# sns.relplot(hab_dat,x='Age',y='Nodes detected',hue='Survival status',style='Survival status',size='Nodes detected')
# plt.show()

accuracy = {'bayes':[],'SVC':[],'svm':[],'k_neighbors':[]}
precision = {'bayes':[],'SVC':[],'svm':[],'k_neighbors':[]}
recall = {'bayes':[],'SVC':[],'svm':[],'k_neighbors':[]}
for i in range(1):
    X_train, X_test, y_train, y_test = train_test_split(hab_dat.iloc[:,:3].values,hab_dat.iloc[:,3].values, test_size=0.20,shuffle=True )
    
    # Naive Bayes
    bayes_clf = GaussianNB()
    bayes_clf.fit(X_train, y_train)
    y_bayes = bayes_clf.predict(X_test)
    accuracy['bayes'].append(met.accuracy_score(y_test,y_bayes))
    precision['bayes'].append(met.precision_score(y_test,y_bayes))
    recall['bayes'].append(met.recall_score(y_test,y_bayes))
    # conf_mat['bayes'] = met.confusion_matrix(y_test,y_bayes)

    # LinearSVC
    SVC_clf = LinearSVC()
    SVC_clf.fit(X_train, y_train)
    y_SVC = SVC_clf.predict(X_test)
    accuracy['SVC'].append(met.accuracy_score(y_test,y_SVC))
    precision['SVC'].append(met.precision_score(y_test,y_SVC))
    recall['SVC'].append(met.recall_score(y_test,y_SVC))
    # conf_mat = met.confusion_matrix(y_test,y_SVC)

    # SVM
    # svm_clf = svm()
    # svm_clf.fit(X_train, y_train)
    # y_svm = svm_clf.predict(X_test)
    # accuracy['svm'].append(met.accuracy_score(y_test,y_svm))
    # precision['svm'].append(met.precision_score(y_test,y_svm))
    # recall['svm'].append(met.recall_score(y_test,y_svm))
    # conf_mat = met.confusion_matrix(y_test,y_svm)

    # K-Neighbors
    k_neighbors_clf = KNeighborsClassifier()
    k_neighbors_clf.fit(X_train, y_train)
    y_k_neighbors = k_neighbors_clf.predict(X_test)
    accuracy['k_neighbors'].append(met.accuracy_score(y_test,y_k_neighbors))
    precision['k_neighbors'].append(met.precision_score(y_test,y_k_neighbors))
    recall['k_neighbors'].append(met.recall_score(y_test,y_k_neighbors))
    # conf_mat = met.confusion_matrix(y_test,y_k_neighbors)
    




# print(conf_mat)
print('accuracy: ', accuracy)
print('precision: ', precision)
print('recall: ', recall)
