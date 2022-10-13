import pandas as pd
import matplotlib.pyplot as plt
import datetime
import time
import numpy as np
import skfuzzy.control as ctrl
import skfuzzy as fuzz
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import cross_validate
from sklearn.neural_network import MLPClassifier
import sklearn.metrics as met
import seaborn as sns
import math

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
    for index in outlier.index:
        data.iloc[index, data.columns == coll ] = data.iloc[index - 1, data.columns == coll ] 
    for index in miss_value.index:
        data.iloc[index, data.columns == coll ] = data.iloc[index - 1, data.columns == coll ]
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


df = pd.read_csv('Proj1_Dataset.csv')

df_clean = df.copy()
df_clean.drop(['Date'], axis=1, inplace=True)
for i in df_clean.columns:    
    if i != 'Time':
        outliers, missing_values = identify_outliers(df_clean, i)
        df_clean = replacing_outliers_missvalues(df_clean, outliers, missing_values, i)

y = df_clean['Persons'].copy()
x = df_clean.drop(columns=['Persons'], axis= 1 , inplace=False)

df_clean['S1Temp'] = (df_clean['S1Temp']+df_clean['S2Temp']+df_clean['S3Temp'])/3
df_clean.rename({'S1Temp': 'STemp'}, axis=1, inplace=True)
df_clean.drop(columns=['STemp','S2Temp','S3Temp'], axis= 1 , inplace=True)

df_clean['S1Light'] = (df_clean['S1Light']+df_clean['S2Light']+df_clean['S3Light'])/3
df_clean.rename({'S1Light': 'SLight'}, axis=1, inplace=True)
df_clean.drop(columns=['S2Light','S3Light'], axis= 1 , inplace=True)

df_clean['CO2'] = np.append(np.diff(df_clean['CO2']),0)
df_clean.rename({'CO2': 'CO2_diff'}, axis=1, inplace=True)
# df_clean.drop(columns=['CO2'], axis= 1 , inplace=True)

df_clean.drop(columns=['PIR1'], axis= 1 , inplace=True)
df_clean.drop(columns=['PIR2'], axis= 1 , inplace=True)

# plt.figure()
# plt.plot(df_clean['CO2_diff'],label='CO2_diff')
# plt.plot(df_clean['SLight'],label='SLight')
# plt.plot(df_clean['Persons'],label='Persons')
# plt.legend()
# plt.show()


Xtrain, Xtest, ytrain, ytest = train_test_split(x,y, test_size=0.10,shuffle=True )

Xtrain, Xval, ytrain, yval = train_test_split(Xtrain,ytrain, test_size=0.10,shuffle=True )


#Correlations

# plt.figure(figsize=(12,10), dpi= 80)
# sns.heatmap(df_clean.corr(), xticklabels=df_clean.corr().columns, yticklabels=df_clean.corr().columns, cmap='RdYlGn', center=0, annot=True)
# plt.title('Correlogram heatmap', fontsize=22)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# # plt.show()

# print(max(df_clean['CO2_diff']))
# print(max(df_clean['SLight']))
# print(min(df_clean['CO2_diff']))
# print(min(df_clean['SLight']))

CO2_variation = ctrl.Antecedent(np.arange(-30, 185, 5),'CO2 variation')
S_light = ctrl.Antecedent(np.arange(0, 468, 1),'Light intensity')
Time = ctrl.Antecedent(np.arange(0, 86400, 1),'Time')

Persons = ctrl.Consequent(np.arange(0,3,0.25),'Number of persons')

# a = np.array(np.arange(0,2880,0.5), dtype='datetime64[s]', datetime.timedelta(seconds=30))
# print(df_clean['CO2_diff'].quantile(0.90))
# print(df_clean['SLight'].quantile(0.92))
# # print(df_clean.index[df_clean['SLight'] == 322])


x = [time.strptime(t,'%H:%M:%S') for t in df_clean['Time']]
s = np.array([datetime.timedelta(hours=t.tm_hour,minutes=t.tm_min,seconds=t.tm_sec).total_seconds() for t in x])
for i, value in enumerate(np.array(s<72000)):
    if value == True:
        s[i] += 14400
    else:
        s[i] -= 72000
df_clean['Time'] = np.array(s)

# print(df_clean.iloc[df_clean.index[df_clean['SLight'] == 322]])
# print(df_clean.loc[(df_clean['SLight'] == 0 )& (df_clean['Persons'] == 3)])
# print(len(df_clean.iloc[df_clean.index[df_clean['Persons'] == 3]]))
# print(min(np.array(df_clean.iloc[df_clean.index[df_clean['Persons'] == 3],df_clean.columns == 'SLight'])))
# print(df_clean.iloc[df_clean.index[df_clean['CO2_diff'] == 5.0]])
# print(min(df_clean['Time']))
# print(min(s))

CO2_variation['decreasing'] = fuzz.trimf(CO2_variation.universe,[-30, -30, -5])
CO2_variation['constant'] = fuzz.trimf(CO2_variation.universe,[-10, 0, 15])
CO2_variation['increasing'] = fuzz.trimf(CO2_variation.universe,[5, 185, 185])

S_light['low'] = fuzz.trimf(S_light.universe,[0, 0, 150])
S_light['medium'] = fuzz.trimf(S_light.universe,[120, 215, 310])
S_light['high'] = fuzz.trimf(S_light.universe,[300, 468, 468])

# S_light['low'] = fuzz.trimf(S_light.universe,[0, 0, 150])
# S_light['medium'] = fuzz.trimf(S_light.universe,[120, 220, 320])
# S_light['high'] = fuzz.trimf(S_light.universe,[310, 468, 468])

Time['night'] = fuzz.trimf(Time.universe,[0,0,43200])
Time['day'] = fuzz.trimf(Time.universe,[39600, 86399, 86399])

Persons['low'] = fuzz.trimf(Persons.universe,[0, 0, 2])
Persons['high'] = fuzz.trimf(Persons.universe,[1, 3, 3])

# Persons['low'] = fuzz.trimf(Persons.universe,[0, 0, 1.25])
# Persons['high'] = fuzz.trimf(Persons.universe,[1, 3, 3])


Persons.view()
Time.view()
CO2_variation.view()
S_light.view()
plt.show()

rule1 = ctrl.Rule(Time['night'] & S_light['low'], Persons['low'])
rule2 = ctrl.Rule(Time['night'] & S_light['medium'], Persons['low'])
rule3 = ctrl.Rule(Time['night'] & S_light['high'], Persons['low'])
rule4 = ctrl.Rule(Time['night'] & CO2_variation['decreasing'], Persons['low'])
rule5 = ctrl.Rule(Time['night'] & CO2_variation['constant'], Persons['low'])
rule6 = ctrl.Rule(Time['night'] & CO2_variation['increasing'], Persons['low'])
rule7 = ctrl.Rule(Time['day'] & CO2_variation['decreasing'], Persons['low'])
# rule8 = ctrl.Rule(Time['day'] & CO2_variation['constant'], Persons['high']) ## n é possivel concluir
rule9 = ctrl.Rule(Time['day'] & CO2_variation['increasing'], Persons['high'])
rule10 = ctrl.Rule(Time['day'] & S_light['low'], Persons['low'])
rule11 = ctrl.Rule(Time['day'] & S_light['medium'], Persons['high'])
rule12 = ctrl.Rule(Time['day'] & S_light['high'], Persons['high'])
rule13 = ctrl.Rule(CO2_variation['increasing'] & S_light['low'], Persons['high']) 
rule14 = ctrl.Rule(CO2_variation['increasing'] & S_light['medium'], Persons['high'])
rule15 = ctrl.Rule(CO2_variation['increasing'] & S_light['high'], Persons['high'])
rule16 = ctrl.Rule(CO2_variation['constant'] & S_light['low'], Persons['low'])
rule17 = ctrl.Rule(CO2_variation['constant'] & S_light['medium'], Persons['low']) # right
rule18 = ctrl.Rule(CO2_variation['constant'] & S_light['high'], Persons['high']) 
rule19 = ctrl.Rule(CO2_variation['decreasing'] & S_light['low'], Persons['low'])
rule20 = ctrl.Rule(CO2_variation['decreasing'] & S_light['medium'], Persons['low'])
rule21 = ctrl.Rule(CO2_variation['decreasing'] & S_light['high'], Persons['high']) #



# n_people_ctrl = ctrl.ControlSystem([rule1,rule2,rule3,rule4,rule5,rule6,rule7,rule9,rule10,rule11,rule12,rule13,rule14,rule15,rule16,rule17,rule18,rule19,rule20,rule21])
n_people_ctrl = ctrl.ControlSystem([rule1,rule2,rule3,rule4,rule5,rule6,rule7,rule9,rule10,rule11,rule12,rule13,rule14,rule15,rule16,rule17,rule18,rule19,rule20,rule21])

# n_people_ctrl = ctrl.ControlSystem([rule1,rule2,rule3,rule4,rule5,rule6,rule7,rule8,rule9,rule10,rule11,rule12,rule13,rule14,rule15,rule16,rule17,rule18,rule19,rule20,rule21])

number_of_people = ctrl.ControlSystemSimulation(n_people_ctrl)



# predicted_outputs = np.array([])
# for i in range(len(df_clean['Time'])):
#     number_of_people.input['CO2 variation'] = df_clean['CO2_diff'][i] 
#     number_of_people.input['Light intensity'] = df_clean['SLight'][i]
#     number_of_people.input['Time'] = df_clean['Time'][i]
#     number_of_people.compute()
#     predicted_outputs = np.append(predicted_outputs, math.ceil(number_of_people.output['Number of persons']))
#     # predicted_outputs = np.append(predicted_outputs, math.floor(number_of_people.output['Number of persons']))

# y_true = np.array(df_clean['Persons']>2)
# y_predicted =  np.array(predicted_outputs>2)

# print('number of right classified samples: ',metrics.accuracy_score(y_true,y_predicted, normalize = False))
# print('accuracy: ',metrics.accuracy_score(y_true,y_predicted))
# print('precision: ',metrics.precision_score(y_true,y_predicted))
# print('recall: ',metrics.recall_score(y_true,y_predicted))
# print('f1: ',metrics.f1_score(y_true,y_predicted))
# print('confusion matrix: \n',metrics.confusion_matrix(y_true,y_predicted))

