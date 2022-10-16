
from joblib import  load
import pandas as pd
import sys
from sklearn import preprocessing
import time
import sklearn.metrics as met
from sklearn.neural_network import MLPClassifier

##################################### Functions #######################

def identify_outliers(data,  coll):

    upper_limit = data[coll].mean() + 5.4*data[coll].std()
    lower_limit = data[coll].mean() - 5.4*data[coll].std()
    out = data[coll][(data[coll] > upper_limit) | (data[coll] < lower_limit)]
    miss_values = data[data[coll].isnull()]
    
    return out, miss_values

def replacing_outliers_missvalues(data, outlier,miss_value, coll):

    for index in outlier.index:
        data.iloc[index, data.columns == coll ] = data.iloc[index - 1, data.columns == coll] 
    for index in miss_value.index:
        data.iloc[index, data.columns == coll ] = data.iloc[index - 1, data.columns == coll]

    return data

def normalize_data(X):
    
    scaler = preprocessing.StandardScaler()
    scaler.fit(X)
    X_data = scaler.transform(X)

    return X_data

def predict(X, Y, model):

    start_time1 = time.process_time()
    y_pred = model.predict(X)
    Time = time.process_time() - start_time1)
    accuracy = append(met.accuracy_score(Y,y_pred))
    precision['model 1'].append(met.precision_score(Y, y_pred,  average=None))
    recall['model 1'].append(met.recall_score(Y,y_pred,  average=None))
    C_matrix['model 1'].append(met.confusion_matrix(Y,y_pred))

    return
##################################### MAIN ############################

if len(sys.argv) < 2:
    print(" ----- Error: arguments should follow the following setup ----- ")
    print("(python) TestMe.py xxx.csv")
    exit(0)
elif sys.argv[1][int(len(sys.argv[1])-4):int(len(sys.argv[1]))] != '.csv':
        print(" ----- Error: arguments should follow the following setup ----- ")
        print("(python) TestMe.py xxx.csv")
else:
    df = pd.read_csv(sys.argv[1])

model_multi = load('Model_Multi_class.joblib') 
model_binary = load('Model_Binary.joblib')

#################################### Outliers, Missing Data and Feature Selection ###############

df_clean = df.copy()
df_clean.drop(['Date', 'Time', 'CO2', 'S3Temp'] , axis=1, inplace=True)

for i in df_clean.columns:
     
    outliers, missing_values = identify_outliers(df_clean, i)
    df_clean = replacing_outliers_missvalues(df_clean, outliers, missing_values, i)

####################################### X and Y data ##########################################

y = df_clean['Persons'].copy()
x = df_clean.drop(columns=['Persons'], axis= 1 , inplace=False)

#######################################  NORMALIZE DATA   #####################################

x_norm_test, x_norm_train = normalize_data(x)