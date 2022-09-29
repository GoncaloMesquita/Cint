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

accuracy = []
precision = []
recall = []
for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(hab_dat.iloc[:,:3].values,hab_dat.iloc[:,3].values, test_size=0.20,shuffle=True )
    
    # Naive Bayes

    bayes_clf = GaussianNB()
    bayes_clf.fit(X_train, y_train)
    y_pred = bayes_clf.predict(X_test)
    accuracy.append(met.accuracy_score(y_test,y_pred))
    precision.append(met.precision_score(y_test,y_pred))
    recall.append(met.recall_score(y_test,y_pred))
    conf_mat = met.confusion_matrix(y_test,y_pred)




print(conf_mat)
print('accuracy: ', mean(accuracy))
print('precision: ', mean(precision))
print('recall: ', mean(recall))
