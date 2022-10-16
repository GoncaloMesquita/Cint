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

    model = MLPClassifier(alpha=0.002, hidden_layer_sizes=(5,), learning_rate_init=0.003)
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
# Fuzzy_classifier(Xtrain, ytrain, number_of_people) # Results with training set

Fuzzy_classifier(Xtest, ytest, number_of_people) # Results with holdout set

############### Hyperparameters tunning ###############

# print(hyperparameters_tunning(Xtrain, ytrain)) # Hyperparameters used in NN classifier

############### NN Classifier ###############

NN_classifier(Xtrain, Xtest, ytrain, ytest)

