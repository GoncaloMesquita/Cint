import sklearn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Haberman

hab_dat = pd.read_table('haberman.data',sep=',',header=None)
hab_dat.columns = ['Age', 'Year of operation', 'Nodes detected', 'Survival status']

# Iris
iris_dat = pd.read_table('iris.data',sep=',',header=None)
iris_dat.columns = ['sepal length','sepal width','petal length','petal length','class']


sns.set_theme(style="darkgrid")
sns.relplot(iris_dat,x='sepal length',y='sepal width',hue='class',style='class')
plt.show()