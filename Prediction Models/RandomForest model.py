# Databricks notebook source
# MAGIC %md
# MAGIC ###Random Forest

# COMMAND ----------

# MAGIC %md
# MAGIC Random Forest is a popular and effective ensemble machine learning algorithm. It is widely used for classification and regression predictive modeling problems with structured (tabular) data sets, e.g. data as it looks in a spreadsheet or database table.
# MAGIC
# MAGIC Random Forest can also be used for **time series forecasting**, although it requires the time series dataset to be transformed into a supervised learning problem first.

# COMMAND ----------

# MAGIC %md
# MAGIC ###Parameters 

# COMMAND ----------

import configparser

config = configparser.ConfigParser()
files_read = config.read('/Workspace/.../config.ini')
country = config['parameters']['country']
min_rotation = int(config['parameters']['min_rotation'])
print(f"Country: {country}")

# COMMAND ----------

# MAGIC %md
# MAGIC ###Load and transform the data

# COMMAND ----------

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
%matplotlib inline

#Load the data from DATA_{country}_Filtered
df = spark.sql(f"""
SELECT * 
FROM DATA_{country}_Filtered  
""")
df = df.toPandas()

#Transform categorical features into numerical one
df['is_stock_numeric'] = df['is_stock'].map({'Y': 1, 'N': 0})
df['FR_numeric'] = df['Forte_rotation'].map({True: 1, False: 0})
df['etim_class_numeric'] = df['etim_class_id'].str[2:8]

#Drop categorical features
df.drop(['avg_sales_price', 'total_discount_value', 'is_stock', 'Forte_rotation', 'sis_sub_family', 'etim_class_id', 'etim_class'], axis=1, inplace=True)
melt = df
melt = melt.sort_values(['First_day', 'product_id'])
melt.reset_index(inplace = True)

#Split the data into train and validation dataset
split_point = '2022-01-03'
melt_train = melt[melt['First_day'] < split_point].copy()
melt_valid = melt[melt['First_day'] >= split_point].copy()

#Set up 1-step target
melt_train['total_qty_next_week'] = melt_train.groupby("product_id")['total_qty'].shift(-1)
melt_train = melt_train.dropna()
melt_valid['total_qty_next_week'] = melt_valid.groupby("product_id")['total_qty'].shift(-1)

#Create 4 fundamental features
#lag_total_qty_1 is the total_qty of the previous date
melt_train["lag_total_qty_1"] = melt_train.groupby("product_id")['total_qty'].shift(1)
melt_valid["lag_total_qty_1"] = melt_valid.groupby("product_id")['total_qty'].shift(1)
#Difference
#diff_total_qty_1 is the difference between the actual value of total_qty and the value from the previous date
melt_train["diff_total_qty_1"] = melt_train.groupby("product_id")['total_qty'].diff(1)
melt_valid["diff_total_qty_1"] = melt_valid.groupby("product_id")['total_qty'].diff(1)
#Rolling stats
#["mean_tqs_4" is the mean of total_qty values from the 4 previous dates
melt_train["mean_tqs_4"] = melt_train.groupby("product_id")['total_qty'].rolling(4).mean().reset_index(level=0, drop=True)
melt_valid["mean_tqs_4"] = melt_valid.groupby("product_id")['total_qty'].rolling(4).mean().reset_index(level=0, drop=True)

#Evaluation metric
def wmape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))
def rmse(y_true, y_pred):
    res = np.sum((y_true - y_pred)**2)
    n = len(y_true)
    return(np.sqrt(res/n))

#Specify the input features of the ML model
features = [
 'total_qty',
 'sales_date_weekly_number',
 'sales_week',
 'sales_year',
 'sis_sub_family_id',
 'sis_group_id',
 'vendor_id',
 'is_stock_numeric',
 'FR_numeric',
 'etim_class_numeric',
 'lag_total_qty_1',
 'diff_total_qty_1',
 'mean_tqs_4']

# COMMAND ----------

# MAGIC %md
# MAGIC ###Hyperparameter tuning

# COMMAND ----------

from sklearn.model_selection import ParameterGrid
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
import itertools
import numpy as np
from pprint import pprint 

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 200, num = 4)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 50, num = 5)]
max_depth.append(None)
# Create the random grid
grid = {'n_estimators': n_estimators,
        'max_features': max_features,
        'max_depth': max_depth,}
pprint(grid)

#Prepare the data for training
imputer = SimpleImputer()
Xtr = imputer.fit_transform(melt_train[features])
ytr = melt_train['total_qty_next_week']
Xval = imputer.transform(melt_valid[features])
yval = melt_valid['total_qty_next_week']

#Start tuning
param_grid = list(itertools.product(n_estimators, max_features))

best_score = float('inf')
best_params = None

for params in param_grid:
    n_est, max_feat = params
    model = RandomForestRegressor(n_estimators=n_est, max_features=max_feat, random_state=0, n_jobs=6)
    model.fit(Xtr, ytr)
    p = model.predict(Xval)
    l = len(p) - len(melt.product_id.unique())
    prediction = p[:l]
    yvalidation = yval[:l]
    score = rmse(yvalidation, prediction)
    if score < best_score:
        best_score = score
        best_params = params
print(best_score, best_params)        

# COMMAND ----------

# MAGIC %md
# MAGIC ###Train a model per group_id

# COMMAND ----------

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
%matplotlib inline

#Load data from the prepared DATA_{country}_Filtered table
df = spark.sql(f"""
SELECT *
FROM DATA_{country}_Filtered                   
""")
df =df.toPandas()

df = df.sort_values(['product_id', 'First_day'])
groups = df.sis_group_id.unique()

e_rmse_group = {}
e_wmape_group = {}

#Evaluation metric
def rmse(y_true, y_pred):
    res = np.sum((y_true - y_pred)**2)
    n = len(y_true)
    return(np.sqrt(res/n))

def wmape(ytrue, ypred):
    return np.sum(np.abs(ytrue - ypred)) / np.sum(np.abs(ytrue))    

#Calculate errors per group : This loop trains the model for each of the sub-datasets, 
for group in groups:
    #Select the rows
    df_group = df[df['sis_group_id'] == group]

    #Transform categorical features
    df_group['is_stock_numeric'] = df_group['is_stock'].map({'Y': 1, 'N': 0})
    df_group['FR_numeric'] = df_group['Forte_rotation'].map({True: 1, False: 0})
    df_group['etim_class_numeric'] = df_group['etim_class_id'].str[2:8]

    #scale the data
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler = scaler.fit(df_group[['total_qty']])
    df_group['total_qty'] = scaler.transform(df_group[['total_qty']])

    #Delete unuseful features
    df_group.drop(['avg_sales_price', 'total_discount_value', 'is_stock', 'Forte_rotation', 'sis_sub_family', 'etim_class_id', 'etim_class'], axis=1, inplace=True)
    melt = df_group
    melt = melt.sort_values(['First_day', 'product_id'])

    #Split the data
    split_point = '2022-01-03'
    melt_train = melt[melt['First_day'] < split_point].copy()
    melt_valid = melt[melt['First_day'] >= split_point].copy()

    #Specify the steps we want to predict
    steps = [1, 2, 3, 4, 12, 26]

    #Create fundamental features
    #lag_total_qty_i est la total_qty de la date précédente 
    for step in steps:
        melt_train[f'lag_total_qty_{step}'] = melt_train.groupby("product_id")['total_qty'].shift(step)
        melt_valid[f'lag_total_qty_{step}'] = melt_valid.groupby("product_id")['total_qty'].shift(step)

    #Difference
    #diff_total_qty_i est la différence avec la total_qty de la date i précédente 
    for step in steps:
        melt_train[f'diff_total_qty_{step}'] = melt_train.groupby("product_id")['total_qty'].diff(step)
        melt_valid[f'diff_total_qty_{step}'] = melt_valid.groupby("product_id")['total_qty'].diff(step)

    #Rolling stats
    #["mean_tqs_i" est la moyenne des total_qty des i dernières dates
    for step in steps[-3:]:
        melt_train[f'mean_tqs_{step}'] = melt_train.groupby("product_id")['total_qty'].rolling(step).mean().reset_index(level=0, drop=True)
        melt_valid[f'mean_tqs_{step}'] = melt_valid.groupby("product_id")['total_qty'].rolling(step).mean().reset_index(level=0, drop=True)

    #target
    for step in steps : 
        melt_train[f'total_qty_next_week_{step}'] = melt_train.groupby("product_id")['total_qty'].shift(-step)
        melt_train = melt_train.dropna()
        melt_valid[f'total_qty_next_week_{step}'] = melt_valid.groupby("product_id")['total_qty'].shift(-step)

    #Bseline model : Predict next week sales as equal to this week sales
    y_pred = scaler.inverse_transform(melt_valid[['total_qty']])
    y_true = scaler.inverse_transform(melt_valid[['total_qty_next_week_1']])
    l = len(y_pred) - len(melt.product_id.unique())
    y_pred = y_pred[:l]
    y_true = y_true[:l]
    #print(rmse(y_true, y_pred))

    #Train the model
    features = melt_train.columns[2:][:-6]
    target = melt_train.columns[-6:]
    imputer = SimpleImputer()
    Xtr = imputer.fit_transform(melt_train[features])
    ytr = melt_train[target]
    mdl = RandomForestRegressor(n_estimators=100, max_features='sqrt', random_state=0, n_jobs=8)
    mdl.fit(Xtr, ytr)

    #Evaluate the model
    Xval = imputer.transform(melt_valid[features])
    yval= {}

    for step in steps:
        yval[f'yval_{step}'] = scaler.inverse_transform(melt_valid[[f'total_qty_next_week_{step}']])
        yval[f'yval_{step}'] = pd.DataFrame(yval[f'yval_{step}'], index= melt_valid.index, columns=melt_valid[[f'total_qty_next_week_{step}']].columns)

    p = mdl.predict(Xval)
    p = pd.DataFrame(p, index= melt_valid.index, columns=target)
    p = scaler.inverse_transform(p)
    p = pd.DataFrame(p, index= melt_valid.index, columns=target)
    
    #Save the error values: e_rmse and e_wmape is a dictionary that contains erros values for the different steps in a certain group_id
    e_rmse = {}
    e_wmape = {}

    for step in steps:
        l = len(p) - step*len(melt.product_id.unique())
        e_rmse[step] = rmse(yval[f'yval_{step}'][f'total_qty_next_week_{step}'][:l], p[f'total_qty_next_week_{step}'][:l])
        e_wmape[step] = wmape(yval[f'yval_{step}'][f'total_qty_next_week_{step}'][:l], p[f'total_qty_next_week_{step}'][:l])
    
    e_rmse_group[group] = e_rmse
    e_wmape_group[group] = e_wmape

d_rmse = pd.DataFrame(e_rmse_group)
d_wmape = pd.DataFrame(e_wmape_group)


# COMMAND ----------

#Measure a total error (for the different steps)
df = df.sort_values(['product_id', 'First_day'])
#Transform categorical features
df['is_stock_numeric'] = df['is_stock'].map({'Y': 1, 'N': 0})
df['FR_numeric'] = df['Forte_rotation'].map({True: 1, False: 0})
df['etim_class_numeric'] = df['etim_class_id'].str[2:8]

#scale the data
scaler = StandardScaler(with_mean=True, with_std=True)
scaler = scaler.fit(df[['total_qty']])
df['total_qty'] = scaler.transform(df[['total_qty']])

#melt the data
df.drop(['avg_sales_price', 'total_discount_value', 'is_stock', 'Forte_rotation', 'sis_sub_family', 'etim_class_id', 'etim_class'], axis=1, inplace=True)
melt = df
melt = melt.sort_values(['First_day', 'product_id'])

#Split the data
split_point = '2022-01-03'
melt_train = melt[melt['First_day'] < split_point].copy()
melt_valid = melt[melt['First_day'] >= split_point].copy()

steps = [1, 2, 3, 4, 12, 26]

#Create fundamental features
#lag_total_qty_i est la total_qty de la date précédente 
for step in steps:
    melt_train[f'lag_total_qty_{step}'] = melt_train.groupby("product_id")['total_qty'].shift(step)
    melt_valid[f'lag_total_qty_{step}'] = melt_valid.groupby("product_id")['total_qty'].shift(step)

#Difference
#diff_total_qty_i est la différence avec la total_qty de la date i précédente 
for step in steps:
    melt_train[f'diff_total_qty_{step}'] = melt_train.groupby("product_id")['total_qty'].diff(step)
    melt_valid[f'diff_total_qty_{step}'] = melt_valid.groupby("product_id")['total_qty'].diff(step)

#Rolling stats
#["mean_tqs_i" est la moyenne des total_qty des i dernières dates
for step in steps[-3:]:
    melt_train[f'mean_tqs_{step}'] = melt_train.groupby("product_id")['total_qty'].rolling(step).mean().reset_index(level=0, drop=True)
    melt_valid[f'mean_tqs_{step}'] = melt_valid.groupby("product_id")['total_qty'].rolling(step).mean().reset_index(level=0, drop=True)

#target
for step in steps : 
    melt_train[f'total_qty_next_week_{step}'] = melt_train.groupby("product_id")['total_qty'].shift(-step)
    melt_train = melt_train.dropna()
    melt_valid[f'total_qty_next_week_{step}'] = melt_valid.groupby("product_id")['total_qty'].shift(-step)

#Train the model
features = melt_train.columns[2:][:-6]
target = melt_train.columns[-6:]
imputer = SimpleImputer()
Xtr = imputer.fit_transform(melt_train[features])
ytr = melt_train[target]
mdl = RandomForestRegressor(n_estimators=100, max_features='sqrt', random_state=0, n_jobs=8)
mdl.fit(Xtr, ytr)

#Evaluate the model
Xval = imputer.transform(melt_valid[features])
yval= {}
for step in steps:
    yval[f'yval_{step}'] = scaler.inverse_transform(melt_valid[[f'total_qty_next_week_{step}']])
    yval[f'yval_{step}'] = pd.DataFrame(yval[f'yval_{step}'], index= melt_valid.index, columns=melt_valid[[f'total_qty_next_week_{step}']].columns)
p = mdl.predict(Xval)
p = pd.DataFrame(p, index= melt_valid.index, columns=target)
p = scaler.inverse_transform(p)
p = pd.DataFrame(p, index= melt_valid.index, columns=target)

#Calculate RMSE and WMAPE for eaxh of the steps and save the erros in the e_wmape and e_rmse dictionaries
e_rmse = {}
e_wmape = {}
for step in steps:
    l = len(p) - step*len(melt.product_id.unique())

    e_rmse[step] = rmse(yval[f'yval_{step}'][f'total_qty_next_week_{step}'][:l], p[f'total_qty_next_week_{step}'][:l])
    e_wmape[step] = wmape(yval[f'yval_{step}'][f'total_qty_next_week_{step}'][:l], p[f'total_qty_next_week_{step}'][:l])

# COMMAND ----------

#Now let's save the measured errors ( Total and per group_id) in 2 SQL tables , one for WMAPE and one for RMSE, to use them to compare the models' performances later.

#d_wmape['total'] = list(e_wmape.values())
#d_rmse['total'] = list(e_rmse.values())

spark_df = spark.createDataFrame(d_rmse)
spark_df.write.mode("overwrite").saveAsTable(f"RMSE_RF_{country}")

spark_df = spark.createDataFrame(d_wmape)
spark_df.write.mode("overwrite").saveAsTable(f"WMAPE_RF_{country}")
