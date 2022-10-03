import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score


df_haberman = pd.read_csv('haberman.data', header=None)
df_haberman.columns = ['Age', 'Year of operation', 'Nodes detected', 'Survival status']
df_iris = pd.read_csv('iris.data', header= None)
df_iris.columns = ['sepal length','sepal width', 'petal length', 'petal width','class' ]

sns.relplot(data=df_iris, x='sepal length', y="sepal width", hue="class", size='sepal length')
plt.show()

sns.relplot(data= df_haberman, x='Age', y='Nodes detected', hue='Survival status', size='Survival status')
plt.show()

Y_data = df_iris['class'].copy()
X_data = df_iris.drop(columns=['class'], axis= 1 , inplace=False)

x_train, x_test, y_train, y_test = train_test_split( X_data, Y_data, test_size=0.30, random_state=42)
# Naive Bayes
bayes_model = GaussianNB()
y_pred = bayes_model.fit(x_train, y_train).predict(x_test)
print(precision_score(y_test, y_pred, average='macro'))

