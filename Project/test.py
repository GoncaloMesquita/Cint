# from random import shuffle
# import pandas as pd
# import matplotlib.pyplot as plt
# import datetime
# import time
# import numpy as np
# import skfuzzy.control as ctrl
# import skfuzzy as fuzz
# from sklearn.model_selection import train_test_split
# from sklearn import metrics
# from sklearn import preprocessing
# from sklearn.model_selection import cross_validate
# from sklearn.neural_network import MLPClassifier
# import sklearn.metrics as met
# import seaborn as sns
# import math

# from sklearn.model_selection import GridSearchCV

# #---------------------Functions------------------------------

# def data_visualization(data, data_clean, coll):
#     #print(df.describe())
#     """ data[coll].plot()
#     plt.title(coll)
#     plt.show() """
#     fig, axs = plt.subplots(2)
#     axs[0].plot(data[coll])
#     axs[1].plot(data_clean[coll])
#     #fig.title(coll)
#     plt.show()
#     return

# def identify_outliers(data,  coll):

#     upper_limit = data[coll].mean() + 5.4*data[coll].std()
#     lower_limit = data[coll].mean() - 5.4*data[coll].std()
#     out = data[coll][(data[coll] > upper_limit) | (data[coll] < lower_limit)]
#     miss_values = data[data[coll].isnull()]
#     return out, miss_values

# def replacing_outliers_missvalues(data, outlier,miss_value, coll):
#     for index in outlier.index:
#         data.iloc[index, data.columns == coll ] = data.iloc[index - 1, data.columns == coll ] 
#     for index in miss_value.index:
#         data.iloc[index, data.columns == coll ] = data.iloc[index - 1, data.columns == coll ]
#     #data.iloc[miss_value.index, data.columns == coll ] = mean_miss
#     #data.iloc[outlier.index, data.columns == coll ] = np.nan
#     #data.interpolate(method = 'linear')
#     return data




# df = pd.read_csv('Proj1_Dataset.csv')

# df_clean = df.copy()
# df_clean.drop(['Date'], axis=1, inplace=True)
# for i in df_clean.columns:    
#     if i != 'Time':
#         outliers, missing_values = identify_outliers(df_clean, i)
#         df_clean = replacing_outliers_missvalues(df_clean, outliers, missing_values, i)



# df_clean['S1Temp'] = (df_clean['S1Temp']+df_clean['S2Temp']+df_clean['S3Temp'])/3
# df_clean.rename({'S1Temp': 'STemp'}, axis=1, inplace=True)
# df_clean.drop(columns=['STemp','S2Temp','S3Temp'], axis= 1 , inplace=True)

# df_clean['S1Light'] = (df_clean['S1Light']+df_clean['S2Light']+df_clean['S3Light'])/3
# df_clean.rename({'S1Light': 'SLight'}, axis=1, inplace=True)
# df_clean.drop(columns=['S2Light','S3Light'], axis= 1 , inplace=True)

# df_clean['CO2'] = np.append(np.diff(df_clean['CO2']),0)
# df_clean.rename({'CO2': 'CO2_diff'}, axis=1, inplace=True)
# # df_clean.drop(columns=['CO2'], axis= 1 , inplace=True)

# df_clean.drop(columns=['PIR1'], axis= 1 , inplace=True)
# df_clean.drop(columns=['PIR2'], axis= 1 , inplace=True)

# # plt.figure()
# # plt.plot(df_clean['CO2_diff'],label='CO2_diff')
# # plt.plot(df_clean['SLight'],label='SLight')
# # plt.plot(df_clean['Persons'],label='Persons')
# # plt.legend()
# # plt.show()



# time_aux = [time.strptime(t,'%H:%M:%S') for t in df_clean['Time']]
# time_aux2 = np.array([datetime.timedelta(hours=t.tm_hour,minutes=t.tm_min,seconds=t.tm_sec).total_seconds() for t in time_aux])
# for i, value in enumerate(np.array(time_aux2<72000)):
#     if value == True:
#         time_aux2[i] += 14400
#     else:
#         time_aux2[i] -= 72000
# df_clean['Time'] = np.array(time_aux2)

# y = df_clean['Persons'].copy()
# x = df_clean.drop(columns=['Persons'], axis= 1 , inplace=False)


# Xtrain, Xtest, ytrain, ytest = train_test_split(x,y, test_size=0.10,random_state=25)
# # Xtrain, Xtest, ytrain, ytest = train_test_split(x,y, test_size=0.10,shuffle=True )




# #Correlations

# # plt.figure(figsize=(12,10), dpi= 80)
# # sns.heatmap(df_clean.corr(), xticklabels=df_clean.corr().columns, yticklabels=df_clean.corr().columns, cmap='RdYlGn', center=0, annot=True)
# # plt.title('Correlogram heatmap', fontsize=22)
# # plt.xticks(fontsize=12)
# # plt.yticks(fontsize=12)
# # plt.show()

# # print(max(df_clean['CO2_diff']))
# # print(max(df_clean['SLight']))
# # print(min(df_clean['CO2_diff']))
# # print(min(df_clean['SLight']))

# CO2_variation = ctrl.Antecedent(np.arange(-30, 185+5, 5),'CO2 variation')
# S_light = ctrl.Antecedent(np.arange(0, 468+1, 1),'Light intensity')
# Time = ctrl.Antecedent(np.arange(0, 86400+1, 1),'Time')

# # CO2_variation = ctrl.Antecedent(np.arange(-30, 185, 2.5),'CO2 variation')
# # S_light = ctrl.Antecedent(np.arange(0, 468, 1),'Light intensity')
# # Time = ctrl.Antecedent(np.arange(0, 86400, 1),'Time')

# # Persons = ctrl.Consequent(np.arange(0,3+0.25,0.25),'Number of persons')
# Persons = ctrl.Consequent(np.arange(0,3+1,1),'Number of persons')

# # Persons = ctrl.Consequent(np.arange(0,3+1,1),'Number of persons','som')
# # Persons = ctrl.Consequent(np.arange(0,3+1,1),'Number of persons','mom')
# # Persons = ctrl.Consequent(np.arange(0,3+1,1),'Number of persons','lom')
# # Persons = ctrl.Consequent(np.arange(0,3+1,1),'Number of persons','bisector')
# # print(Persons.defuzzify_method)
# # a = np.array(np.arange(0,2880,0.5), dtype='datetime64[s]', datetime.timedelta(seconds=30))
# # print(df_clean['CO2_diff'].quantile(0.90))
# # print(df_clean['SLight'].quantile(0.33*2))
# # # print(df_clean.index[df_clean['SLight'] == 322])




# # print(df_clean.iloc[df_clean.index[df_clean['SLight'] == 322]])
# # print(df_clean.loc[(df_clean['SLight'] == 0 )& (df_clean['Persons'] == 3)])
# # print(len(df_clean.iloc[df_clean.index[df_clean['Persons'] == 3]]))
# # print(min(np.array(df_clean.iloc[df_clean.index[df_clean['Persons'] == 3],df_clean.columns == 'SLight'])))
# # print(df_clean.iloc[df_clean.index[df_clean['CO2_diff'] == 5.0]])
# # print(min(df_clean['Time']))
# # print(min(s))

# # CO2_variation['decreasing'] = fuzz.trimf(CO2_variation.universe,[-30, -30, -5])
# # CO2_variation['constant'] = fuzz.trimf(CO2_variation.universe,[-10, 0, 15])
# # CO2_variation['increasing'] = fuzz.trimf(CO2_variation.universe,[5, 185, 185])

# CO2_variation['decreasing'] = fuzz.trapmf(CO2_variation.universe,[-30, -30, -15, -5])
# CO2_variation['constant'] = fuzz.trimf(CO2_variation.universe,[-15, 0, 15])
# CO2_variation['increasing'] = fuzz.trapmf(CO2_variation.universe,[5, 15, 185, 185])

# # S_light['low'] = fuzz.trimf(S_light.universe,[0, 0, 150])
# # S_light['medium'] = fuzz.trimf(S_light.universe,[120, 215, 340])
# # S_light['high'] = fuzz.trimf(S_light.universe,[290, 468, 468])

# S_light['low'] = fuzz.trapmf(S_light.universe,[0, 0, 60, 90])
# S_light['medium'] = fuzz.trimf(S_light.universe,[80, 215, 340])
# S_light['high'] = fuzz.trimf(S_light.universe,[290, 468, 468])

# # sns.set_theme(style="darkgrid")
# # # sns.relplot(df_clean,x='Persons',y='CO2_diff')

# # sns.relplot(df_clean,x='Persons',y='SLight',hue='Time', size= 'CO2_diff')


# # plt.show()

# # S_light['low'] = fuzz.trapmf(S_light.universe,[0, 0, 40, 60])
# # S_light['medium'] = fuzz.trapmf(S_light.universe,[5, 100, 320, 468])
# # S_light['high'] = fuzz.trapmf(S_light.universe,[90, 200, 468, 468])


# # Time['night'] = fuzz.trapmf(Time.universe,[0,43200*0.25,43200*0.75,43200])
# # Time['day'] = fuzz.trapmf(Time.universe,[39600, 39600*1.2, 86399*0.85,86399])

# Time['night'] = fuzz.trapmf(Time.universe,[0,43200*0.25,43200*0.75,43200])
# Time['morning'] = fuzz.trapmf(Time.universe,[39600, 39600*1.1, 64800*0.9, 64800])
# Time['afternoon'] = fuzz.trapmf(Time.universe,[61200, 64800, 86400*0.9,86400])


# Persons['low'] = fuzz.trimf(Persons.universe,[0, 0, 2])
# Persons['high'] = fuzz.trimf(Persons.universe,[1, 3, 3])


# # Persons['low'] = fuzz.trapmf(Persons.universe,[0, 0, 1, 1])
# # Persons['high'] = fuzz.trapmf(Persons.universe,[0, 3, 3, 3])


# # Persons.view()
# # Time.view()
# # CO2_variation.view()
# # S_light.view()
# # plt.show()

# # rule1 = ctrl.Rule(Time['night'] & S_light['low'], Persons['low'])
# # rule2 = ctrl.Rule(Time['night'] & S_light['medium'], Persons['low'])
# # rule3 = ctrl.Rule(Time['night'] & S_light['high'], Persons['low'])
# # rule4 = ctrl.Rule(Time['night'] & CO2_variation['decreasing'], Persons['low'])
# # rule5 = ctrl.Rule(Time['night'] & CO2_variation['constant'], Persons['low'])
# # rule6 = ctrl.Rule(Time['night'] & CO2_variation['increasing'], Persons['low'])
# # rule7 = ctrl.Rule(Time['day'] & CO2_variation['decreasing'], Persons['low']) #
# # rule8 = ctrl.Rule(Time['day'] & CO2_variation['constant'], Persons['high']) ## não é possivel concluir 
# # rule9 = ctrl.Rule(Time['day'] & CO2_variation['increasing'], Persons['high'])
# # rule10 = ctrl.Rule(Time['day'] & S_light['low'], Persons['low']) #
# # rule11 = ctrl.Rule(Time['day'] & S_light['medium'], Persons['low']) #
# # rule12 = ctrl.Rule(Time['day'] & S_light['high'], Persons['high'])
# # rule13 = ctrl.Rule(CO2_variation['increasing'] & S_light['low'], Persons['high']) #
# # rule14 = ctrl.Rule(CO2_variation['increasing'] & S_light['medium'], Persons['high']) #
# # rule15 = ctrl.Rule(CO2_variation['increasing'] & S_light['high'], Persons['high'])
# # rule16 = ctrl.Rule(CO2_variation['constant'] & S_light['low'], Persons['low'])
# # rule17 = ctrl.Rule(CO2_variation['constant'] & S_light['medium'], Persons['low']) # right
# # rule18 = ctrl.Rule(CO2_variation['constant'] & S_light['high'], Persons['high']) 
# # rule19 = ctrl.Rule(CO2_variation['decreasing'] & S_light['low'], Persons['low'])
# # rule20 = ctrl.Rule(CO2_variation['decreasing'] & S_light['medium'], Persons['high'])
# # rule21 = ctrl.Rule(CO2_variation['decreasing'] & S_light['high'], Persons['high']) #

# rule1 = ctrl.Rule(Time['night'] & S_light['low'], Persons['low'])
# rule2 = ctrl.Rule(Time['night'] & S_light['medium'], Persons['low'])
# rule3 = ctrl.Rule(Time['night'] & S_light['high'], Persons['low'])
# rule4 = ctrl.Rule(Time['night'] & CO2_variation['decreasing'], Persons['low'])
# rule5 = ctrl.Rule(Time['night'] & CO2_variation['constant'], Persons['low'])
# rule6 = ctrl.Rule(Time['night'] & CO2_variation['increasing'], Persons['low'])
# # rule7 = ctrl.Rule(Time['morning'] & CO2_variation['decreasing'], Persons['low']) #
# # rule8 = ctrl.Rule(Time['morning'] & CO2_variation['constant'], Persons['high']) ## não é possivel concluir 
# # rule9 = ctrl.Rule(Time['morning'] & CO2_variation['increasing'], Persons['high'])
# rule10 = ctrl.Rule(Time['morning'] & S_light['low'], Persons['low']) 
# rule11 = ctrl.Rule(Time['morning'] & S_light['medium'], Persons['low']) 
# rule12 = ctrl.Rule(Time['morning'] & S_light['high'], Persons['high'])
# # rule13 = ctrl.Rule(Time['afternoon'] & CO2_variation['decreasing'], Persons['low']) #
# # rule14 = ctrl.Rule(Time['afternoon'] & CO2_variation['constant'], Persons['high']) ## não é possivel concluir 
# # rule15 = ctrl.Rule(Time['afternoon'] & CO2_variation['increasing'], Persons['high'])
# rule16 = ctrl.Rule(Time['afternoon'] & S_light['low'], Persons['high']) 
# rule17 = ctrl.Rule(Time['afternoon'] & S_light['medium'], Persons['high']) #
# rule18 = ctrl.Rule(Time['afternoon'] & S_light['high'], Persons['high']) #
# rule19 = ctrl.Rule(CO2_variation['increasing'] & S_light['low'], Persons['high']) 
# rule20 = ctrl.Rule(CO2_variation['increasing'] & S_light['medium'], Persons['high']) 
# rule21 = ctrl.Rule(CO2_variation['increasing'] & S_light['high'], Persons['high'])
# rule22 = ctrl.Rule(CO2_variation['constant'] & S_light['low'], Persons['low']) #
# rule23 = ctrl.Rule(CO2_variation['constant'] & S_light['medium'], Persons['low']) #
# rule24 = ctrl.Rule(CO2_variation['constant'] & S_light['high'], Persons['high']) #
# rule25 = ctrl.Rule(CO2_variation['decreasing'] & S_light['low'], Persons['low'])
# rule26 = ctrl.Rule(CO2_variation['decreasing'] & S_light['medium'], Persons['high'])
# rule27 = ctrl.Rule(CO2_variation['decreasing'] & S_light['high'], Persons['high']) 



# n_people_ctrl = ctrl.ControlSystem([rule1,rule2,rule3,rule4,rule5,rule6,rule10,rule11,rule12, rule16,rule17,rule18,rule19,rule20,rule21,rule22,rule23,rule24,rule25,rule26,rule27])
# # n_people_ctrl = ctrl.ControlSystem([rule1,rule2,rule3,rule4,rule5,rule6,rule10,rule11,rule12,rule13,rule14,rule15,rule16,rule17,rule18,rule19,rule20,rule21])


# number_of_people = ctrl.ControlSystemSimulation(n_people_ctrl)


# # print(ytrain)

# def Fuzzy_classifier(X,Y,simulation):
#     predicted_outputs = np.array([])
#     predicted_outputs1 = np.array([])

#     X = X.reset_index(drop=True)
#     Y = Y.reset_index(drop=True)
    
#     for i in range(len(X['Time'])):
#         simulation.input['CO2 variation'] = X['CO2_diff'][i] 
#         simulation.input['Light intensity'] = X['SLight'][i]
#         simulation.input['Time'] = X['Time'][i]
#         simulation.compute()
#         predicted_outputs = np.append(predicted_outputs, math.ceil(simulation.output['Number of persons']))
#         predicted_outputs1 = np.append(predicted_outputs1, simulation.output['Number of persons'])
#         # if i == 8470:
#         #     Persons.view(sim = number_of_people)
#         #     Time.view(sim = number_of_people)
#         #     CO2_variation.view(sim = number_of_people)
#         #     S_light.view(sim = number_of_people)
#         #     plt.show()
    
    
#     predicted_outputs1 = pd.DataFrame({'pred':predicted_outputs1})
#     y_true = np.array(Y>2)
#     y_predicted =  np.array(predicted_outputs>2)
#     mask = np.array(y_predicted != y_true)
#     print('Fuzzy number of right classified samples: ',metrics.accuracy_score(y_true,y_predicted, normalize = False))
#     print('Fuzzy accuracy: ',metrics.accuracy_score(y_true,y_predicted))
#     print('Fuzzy precision: ',metrics.precision_score(y_true,y_predicted))
#     print('Fuzzy recall: ',metrics.recall_score(y_true,y_predicted))
#     print('Fuzzy f1: ',metrics.f1_score(y_true,y_predicted))
#     print('Fuzzy confusion matrix: \n',metrics.confusion_matrix(y_true,y_predicted))
#     df = pd.concat([X, Y, predicted_outputs1], axis = 1)
#     df = df[mask]
#     # print(df['Time'].loc[df['Persons'] == 3].mean())
#     # print(df['SLight'].loc[df['Persons'] == 3].max())
#     # print(df.loc[df['Persons'] == 3])
#     # print((df.loc[df['Persons'] == 2]))
#     # print((df.loc[(df['Persons'] == 3) & (df['SLight'] == 28.000000)]))
#     # print((df.loc[(df['Persons'] == 3) & (df['Time'] == 85228)]))
#     # print((df.loc[(df['Persons'] == 2) & (df['CO2_diff'] == 0)& (df['SLight'] <= 150.000000)]))
#     # print((df.loc[(df['Persons'] == 3) & (df['pred'] == 1.5)]))
#     # print((df.loc[df['Persons'] == 3].max()))
#     # print((df.loc[df['Persons'] == 2].max()))


    

# Fuzzy_classifier(Xtrain, ytrain, number_of_people)

# Fuzzy_classifier(Xtest, ytest, number_of_people)


# def hyperparameters_tunning(x_training, y_training):


#     parameters={ 'hidden_layer_sizes':[(6,),(8,), (7,), (5,)],  'alpha':[0.001, 0.002, 0.003, 0.004], 'learning_rate_init':[ 0.003,0.002, 0.001]}
#     scoring = {'recall': met.make_scorer(met.recall_score), 'precision': met.make_scorer(met.precision_score)}
#     # gs_cv = GridSearchCV(MLPClassifier(), parameters, scoring=scoring, cv= 5,refit="recall")
#     gs_cv = GridSearchCV(MLPClassifier(), parameters, scoring=scoring, cv= 5,refit="precision")
#     gs_cv.fit(x_training,y_training)
#     return gs_cv.best_params_

# def NN_classifier(Xtrain, Xtest, ytrain, ytest):
#     ytrain = np.array(ytrain>2)
#     ytest = np.array(ytest>2)
#     # ytest = ytrain

#     # print(hyperparameters_tunning(Xtrain, ytrain))

#     model = MLPClassifier(alpha=0.002, hidden_layer_sizes=(5,), learning_rate_init=0.003)
#     model.fit(Xtrain, ytrain)
#     y_predicted = model.predict(Xtest)
#     # y_predicted = model.predict(Xtrain)

#     print('MLP number of right classified samples: ',metrics.accuracy_score(ytest,y_predicted, normalize = False))
#     print('MLP accuracy: ',metrics.accuracy_score(ytest,y_predicted))
#     print('MLP precision: ',metrics.precision_score(ytest,y_predicted))
#     print('MLP recall: ',metrics.recall_score(ytest,y_predicted))
#     print('MLP f1: ',metrics.f1_score(ytest,y_predicted))
#     print('MLP confusion matrix: \n',metrics.confusion_matrix(ytest,y_predicted))

# # NN_classifier(Xtrain, Xtest, ytrain, ytest)

import pandas as pd
import matplotlib.pyplot as plt
import datetime
import time
import numpy as np
import skfuzzy.control as ctrl
import skfuzzy as fuzz
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
import sklearn.metrics as met
import math
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing

import seaborn as sns
################################################  FUNCTIONS  ################################################

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
    return data

def Fuzzy_classifier(X,Y,simulation):
    predicted_outputs = np.array([])

    X = X.reset_index(drop=True)
    Y = Y.reset_index(drop=True)
    
    for i in range(len(X['Time'])):
        simulation.input['CO2 variation'] = X['CO2_diff'][i] 
        simulation.input['Light intensity'] = X['SLight'][i]
        simulation.input['Time'] = X['Time'][i]
        simulation.compute()
        predicted_outputs = np.append(predicted_outputs, math.ceil(simulation.output['Number of persons']))
    
    y_true = np.array(Y>2)
    y_predicted =  np.array(predicted_outputs>2)
    print('Fuzzy number of right classified samples: ',metrics.accuracy_score(y_true,y_predicted, normalize = False))
    print('Fuzzy accuracy: ',metrics.accuracy_score(y_true,y_predicted))
    print('Fuzzy precision: ',metrics.precision_score(y_true,y_predicted))
    print('Fuzzy recall: ',metrics.recall_score(y_true,y_predicted))
    print('Fuzzy f1: ',metrics.f1_score(y_true,y_predicted))
    print('Fuzzy confusion matrix: \n',metrics.confusion_matrix(y_true,y_predicted))

    sns.relplot(X,x='CO2_diff',y='SLight', hue= y_true.astype(int))
    sns.relplot(X,x='Time',y='SLight', hue= y_true.astype(int))
    plt.show()
def normalize_data(xtrain, xtest):
    
    scaler = preprocessing.StandardScaler()
    scaler.fit(xtrain)
    x_n_train = scaler.transform(xtrain)
    x_n_test = scaler.transform(xtest)

    return x_n_train, x_n_test 
    
def hyperparameters_tunning(x_training, y_training):
    y_training = np.array(y_training>2)

    parameters={ 'hidden_layer_sizes':[(6,),(8,), (7,), (5,)],  'alpha':[0.001, 0.002, 0.003, 0.004], 'learning_rate_init':[ 0.003,0.002, 0.001]}
    scoring = {'recall': met.make_scorer(met.recall_score), 'precision': met.make_scorer(met.precision_score)}
    gs_cv = GridSearchCV(MLPClassifier(), parameters, scoring=scoring, cv= 5,refit="precision")
    gs_cv.fit(x_training,y_training)
    return gs_cv.best_params_

def NN_classifier(Xtrain, Xtest, ytrain, ytest):

    ytrain = np.array(ytrain>2)
    ytest = np.array(ytest>2)

    model = MLPClassifier(alpha=0.001, hidden_layer_sizes=(7,), learning_rate_init=0.001)
    model.fit(Xtrain, ytrain)
    y_predicted = model.predict(Xtest)

    print('MLP number of right classified samples: ',metrics.accuracy_score(ytest,y_predicted, normalize = False))
    print('MLP accuracy: ',metrics.accuracy_score(ytest,y_predicted))
    print('MLP precision: ',metrics.precision_score(ytest,y_predicted))
    print('MLP recall: ',metrics.recall_score(ytest,y_predicted))
    print('MLP f1: ',metrics.f1_score(ytest,y_predicted))
    print('MLP confusion matrix: \n',metrics.confusion_matrix(ytest,y_predicted))

################################################  MAIN  ################################################
df = pd.read_csv('Proj1_Dataset.csv')

############### Outliers, Missing Data and Feature Selection and Transformation ###############

df_clean = df.copy()
df_clean.drop(['Date'], axis=1, inplace=True)

for i in df_clean.columns:    
    if i != 'Time':
        outliers, missing_values = identify_outliers(df_clean, i)
        df_clean = replacing_outliers_missvalues(df_clean, outliers, missing_values, i)


df_clean.drop(columns=['S1Temp','S2Temp','S3Temp'], axis= 1 , inplace=True)

df_clean['S1Light'] = (df_clean['S1Light']+df_clean['S2Light']+df_clean['S3Light'])/3
df_clean.rename({'S1Light': 'SLight'}, axis=1, inplace=True)
df_clean.drop(columns=['S2Light','S3Light'], axis= 1 , inplace=True)

df_clean['CO2'] = np.append(np.diff(df_clean['CO2']),0)
df_clean.rename({'CO2': 'CO2_diff'}, axis=1, inplace=True)

df_clean.drop(columns=['PIR1'], axis= 1 , inplace=True)
df_clean.drop(columns=['PIR2'], axis= 1 , inplace=True)

# Transformation of Time variable into seconds
time_aux = [time.strptime(t,'%H:%M:%S') for t in df_clean['Time']]
time_aux2 = np.array([datetime.timedelta(hours=t.tm_hour,minutes=t.tm_min,seconds=t.tm_sec).total_seconds() for t in time_aux])
for i, value in enumerate(np.array(time_aux2<72000)):
    if value == True:
        time_aux2[i] += 14400
    else:
        time_aux2[i] -= 72000

df_clean['Time'] = np.array(time_aux2)

y = df_clean['Persons'].copy()
x = df_clean.drop(columns=['Persons'], axis= 1 , inplace=False)

############### SPLIT DATA ###############

Xtrain, Xtest, ytrain, ytest = train_test_split(x,y, test_size=0.10,shuffle=True)

############### Fuzzy variables ###############
# Inputs:
CO2_variation = ctrl.Antecedent(np.arange(-30, 185+5, 5),'CO2 variation')
S_light = ctrl.Antecedent(np.arange(0, 468+1, 1),'Light intensity')
Time = ctrl.Antecedent(np.arange(0, 86400+1, 1),'Time')

# Output:
Persons = ctrl.Consequent(np.arange(0,3+1,1),'Number of persons')


# print(Persons.defuzzify_method) # Method used in defuzzification

############### Membership functions ###############

CO2_variation['decreasing'] = fuzz.trapmf(CO2_variation.universe,[-30, -30, -15, -5])
CO2_variation['constant'] = fuzz.trimf(CO2_variation.universe,[-15, 0, 15])
CO2_variation['increasing'] = fuzz.trapmf(CO2_variation.universe,[5, 15, 185, 185])

S_light['low'] = fuzz.trimf(S_light.universe,[0, 0, 150])
S_light['medium'] = fuzz.trimf(S_light.universe,[120, 215, 340])
S_light['high'] = fuzz.trimf(S_light.universe,[290, 468, 468])

Time['night'] = fuzz.trapmf(Time.universe,[0,43200*0.25,43200*0.75,43200])
Time['day'] = fuzz.trapmf(Time.universe,[39600, 39600*1.2, 86400*0.85,86400])

Persons['low'] = fuzz.trimf(Persons.universe,[0, 0, 2])
Persons['high'] = fuzz.trimf(Persons.universe,[1, 3, 3])

############### Visualization of fuzzy variables ###############

# Persons.view()
# Time.view()
# CO2_variation.view()
# S_light.view()
# plt.show()

############### Rules for fuzzy system ###############

rule1 = ctrl.Rule(Time['night'] & S_light['low'], Persons['low'])
rule2 = ctrl.Rule(Time['night'] & S_light['medium'], Persons['low'])
rule3 = ctrl.Rule(Time['night'] & S_light['high'], Persons['low'])
rule4 = ctrl.Rule(Time['night'] & CO2_variation['decreasing'], Persons['low'])
rule5 = ctrl.Rule(Time['night'] & CO2_variation['constant'], Persons['low'])
rule6 = ctrl.Rule(Time['night'] & CO2_variation['increasing'], Persons['low'])
rule7 = ctrl.Rule(Time['day'] & CO2_variation['decreasing'], Persons['low']) # Rule not used due to the worsening of results independently of the output chosen 
rule8 = ctrl.Rule(Time['day'] & CO2_variation['constant'], Persons['high']) # Inconclusive output taking in consideration the antecedents and Rule not used due to the worsening of results independently of the output chosen 
rule9 = ctrl.Rule(Time['day'] & CO2_variation['increasing'], Persons['high']) # Rule not used due to the worsening of results independently of the output chosen 
rule10 = ctrl.Rule(Time['day'] & S_light['low'], Persons['low']) 
rule11 = ctrl.Rule(Time['day'] & S_light['medium'], Persons['low']) 
rule12 = ctrl.Rule(Time['day'] & S_light['high'], Persons['high'])
rule13 = ctrl.Rule(CO2_variation['increasing'] & S_light['low'], Persons['high']) 
rule14 = ctrl.Rule(CO2_variation['increasing'] & S_light['medium'], Persons['high']) 
rule15 = ctrl.Rule(CO2_variation['increasing'] & S_light['high'], Persons['high'])
rule16 = ctrl.Rule(CO2_variation['constant'] & S_light['low'], Persons['low'])
rule17 = ctrl.Rule(CO2_variation['constant'] & S_light['medium'], Persons['low']) 
rule18 = ctrl.Rule(CO2_variation['constant'] & S_light['high'], Persons['high']) 
rule19 = ctrl.Rule(CO2_variation['decreasing'] & S_light['low'], Persons['low'])
rule20 = ctrl.Rule(CO2_variation['decreasing'] & S_light['medium'], Persons['low'])
rule21 = ctrl.Rule(CO2_variation['decreasing'] & S_light['high'], Persons['high']) 

############### Control system and simulation ###############
# Control System that includes all the rules
# n_people_ctrl = ctrl.ControlSystem([rule1,rule2,rule3,rule4,rule5,rule6,rule7,rule8,rule9,rule10,rule11,rule12,rule13,rule14,rule15,rule16,rule17,rule18,rule19,rule20,rule21])

# Control System that obtains best results
n_people_ctrl = ctrl.ControlSystem([rule1,rule2,rule3,rule4,rule5,rule6,rule10,rule11,rule12,rule13,rule14,rule15,rule16,rule17,rule18,rule19,rule20,rule21])

number_of_people = ctrl.ControlSystemSimulation(n_people_ctrl)

############### Fuzzy classifier ###############
Fuzzy_classifier(Xtrain, ytrain, number_of_people) # Results with training set

Fuzzy_classifier(Xtest, ytest, number_of_people) # Results with holdout set

############### Data normalization for NN ###############

Xtrain, Xtest = normalize_data(Xtrain, Xtest)

############### Hyperparameters tunning for NN ###############

# print(hyperparameters_tunning(Xtrain, ytrain)) # Hyperparameters used in NN classifier

############### NN Classifier ###############

NN_classifier(Xtrain, Xtest, ytrain, ytest)

