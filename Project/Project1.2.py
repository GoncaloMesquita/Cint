import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np
import skfuzzy.control as ctrl
import skfuzzy as fuzz
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import cross_validate
from sklearn.neural_network import MLPClassifier
import sklearn.metrics as met
import seaborn as sns
from datetime import datetime, timedelta
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

df_clean['CO2_diff'] = np.append(np.diff(df_clean['CO2']),0)
df_clean.drop(columns=['CO2'], axis= 1 , inplace=True)

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
print(max(df_clean['CO2_diff']))
print(max(df_clean['SLight']))
print(min(df_clean['CO2_diff']))
print(min(df_clean['SLight']))

CO2_variation = ctrl.Antecedent(np.arange(-30, 185, 5),'CO2 variation')
S_light = ctrl.Antecedent(np.arange(0, 468, 1),'Light intensity')
Time = ctrl.Antecedent(np.arange(0, 5, 1),'Time')

Persons = ctrl.Consequent(np.arange(0,3.5,0.5),'Number of persons')

# a = np.array(np.arange(0,2880,0.5), dtype='datetime64[s]', datetime.timedelta(seconds=30))

CO2_variation['drecreasing'] = fuzz.trimf(CO2_variation.universe,[0, 0, 50])
CO2_variation['constant'] = fuzz.trimf(CO2_variation.universe,[30, 50, 70])
CO2_variation['increasing'] = fuzz.trimf(CO2_variation.universe,[50, 100, 100])

S_light['low'] = fuzz.trimf(S_light.universe,[0, 0, 1.5])
S_light['medium'] = fuzz.trimf(S_light.universe,[0.5, 2, 3.5])
S_light['low'] = fuzz.trimf(S_light.universe,[2.5, 4, 4])

Time['night'] = fuzz.trimf(Time.universe,[0, 0, 3500])
Time['day'] = fuzz.trimf(Time.universe,[2500, 6000, 6000])

Persons['low'] = fuzz.trimf(Persons.universe,[0, 0, 2])
Persons['high'] = fuzz.trimf(Persons.universe,[1.5, 3, 3])

# datetimeindex = pd.date_range('2021-10-20 00:00:00', '2021-10-20 23:59:59', freq='s')           
# >>> print(datetimeindex.time[-1])

Persons.view()
# clock_speed.view()
# fan_speed.view()
# plt.show()

# rule1 = ctrl.Rule(core_temp['cold'] & clock_speed['low'], fan_speed['slow'])
# rule2 = ctrl.Rule(core_temp['cold'] & clock_speed['normal'], fan_speed['slow'])
# rule3 = ctrl.Rule(core_temp['cold'] & clock_speed['turbo'], fan_speed['fast'])
# rule4 = ctrl.Rule(core_temp['warm'] & clock_speed['low'], fan_speed['slow'])
# rule5 = ctrl.Rule(core_temp['warm'] & clock_speed['normal'], fan_speed['slow'])
# rule6 = ctrl.Rule(core_temp['warm'] & clock_speed['turbo'], fan_speed['fast'])
# rule7 = ctrl.Rule(core_temp['hot'] & clock_speed['low'], fan_speed['fast'])
# rule8 = ctrl.Rule(core_temp['hot'] & clock_speed['normal'], fan_speed['fast'])
# rule9 = ctrl.Rule(core_temp['hot'] & clock_speed['turbo'], fan_speed['fast'])