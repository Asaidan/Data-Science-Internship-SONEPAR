# Databricks notebook source
# MAGIC %md
# MAGIC ###Long-term short-term memory (LSTM) model

# COMMAND ----------

# MAGIC %md
# MAGIC Machine learning algorithms like **neural networks** and gradient boosting machines have been increasingly employed in multivariate forecasting tasks due to their ability to capture intricate patterns and nonlinear relationships within data. In this notebook, we will be using **Long-term short-term memory (LSTM)**, which represents a major advancement of recurrent neural networks (RNNs) in Deep Learning.

# COMMAND ----------

#Tensorflow installation ( we should execute the cell everytime we oepn this notebook)
%pip install tensorflow

# COMMAND ----------

#Restart the kernel using dbutils.library.restartPython() to use updated packages
dbutils.library.restartPython()

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

#Import packages
import sklearn
import numpy as np
import pandas as pd
import math
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import math

#Load the data from DATA_{country}_Filtered
df = spark.sql(f"""
SELECT *
FROM DATA_{country}_Filtered                   
""")
df_country =df.toPandas()

#Sort and reshape the data
df_country = df_country.sort_values(['First_day', 'product_id'])
df = df_country[['First_day', 'product_id' , 'total_qty']]
df_pivot = df.pivot(index='First_day', columns='product_id', values='total_qty')

#Scale the data using the standization method
scaler = StandardScaler(with_mean=True, with_std=True)
scaler.fit(df_pivot)
df_scaled = scaler.transform(df_pivot)
df_scaled = pd.DataFrame(df_scaled, index=df_pivot.index, columns=df_pivot.columns)
df_pivot = df_scaled

#WMAPE calculation function
def wmape(ytrue, ypred):
    return np.sum(np.abs(ytrue - ypred)) / np.sum(np.abs(ytrue))  
def rmse(y_true, y_pred):
    res = np.sum((y_true - y_pred)**2)
    n = len(y_true)
    return(np.sqrt(res/n))

# COMMAND ----------

# MAGIC %md
# MAGIC ###Create the input and output features of the model

# COMMAND ----------

#predict multiple steps into the future
def multipleStepSampler(df, window):
    xRes = []
    yRes = []
    for i in range(0, len(df) - window - 25):
        res = []
        resultat = []
        for j in range(0, window):
            r = []
            for col in df.columns:
                r.append(df[col][i + j])
            res.append(r)
        xRes.append(res)
        for j in range(0, 26):
            if j in [0, 1, 2, 3, 11, 25]: 
                r = []
                for col in df.columns:
                    r.append(df[col][i + j + window])
                resultat.append(r)
        yRes.append(resultat)
    return np.array(xRes), np.array(yRes)

x, y = multipleStepSampler(df_pivot, 50)  

#Split the data into train and test
SPLIT = 0.85
X_train = x[:int(SPLIT * len(x))]
y_train = y[:int(SPLIT * len(y))]
X_test = x[int(SPLIT * len(x)):]
y_test = y[int(SPLIT * len(y)):]

# COMMAND ----------

# MAGIC %md
# MAGIC ###Train the model

# COMMAND ----------

#Implemet the model
multivariate_lstm = keras.Sequential()
#model.add(InputLayer((window, n_features)))
multivariate_lstm.add(keras.layers.LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2])))    
multivariate_lstm.add(keras.layers.Dropout(0.2))
#6 time steps * n_features( = n_products) per time step
multivariate_lstm.add(keras.layers.Dense(6 * y_train.shape[2], activation='relu')) 
# Reshape output to match y_train shape
multivariate_lstm.add(keras.layers.Reshape((6, y_train.shape[2])))  
multivariate_lstm.compile(loss = 'MeanSquaredError', metrics=['MAE'], optimizer='Adam')

#Fit the model
history = multivariate_lstm.fit(X_train, y_train, epochs= 10, batch_size = 20)
#evaluate the model
predicted_values = multivariate_lstm.predict(X_test[-1:])
  # Reshape predicted_values to 2D array
predicted_values_2d = predicted_values.reshape(-1, predicted_values.shape[-1])
  # Apply inverse transform
predicted_values_inversed = scaler.inverse_transform(predicted_values_2d)
  # Reshape back to (time_steps, features)
pred = predicted_values_inversed.reshape(predicted_values.shape)
  # Same for test
y_test_2d = y_test[-1:].reshape(-1, y_test[-1:].shape[-1])
y_test_inversed = scaler.inverse_transform(y_test_2d)
test = y_test_inversed.reshape(y_test[-1:].shape)
  # Select the specific indices
indices_part1 = df_pivot.index[-26:][0:4]  # First 4 indices
indices_part2 = [df_pivot.index[-26:][11]]  # 12th index, needs to be in a list to concatenate
indices_part3 = [df_pivot.index[-26:][25]]  # Last index, needs to be in a list to concatenate
  # Combine the selected indices
combined_indices = indices_part1.append(pd.to_datetime(indices_part2 + indices_part3))
  # Convert to DatetimeIndex
result_index = pd.DatetimeIndex(combined_indices)
  #pred to df
output_df = pd.DataFrame(pred[0], index=result_index, columns=df_pivot.columns)
  #test to df
test_df = pd.DataFrame(test[0], index=result_index, columns=df_pivot.columns)
  #Calculate WMAPE and RMSE per step
RMSE_step = {}
WMAPE_step = {}
total_RMSE = 0 #For tuning
a = 0 #For tuning
b = 0
for index in test_df.index:
    RMSE_step[index] = np.sqrt(mean_squared_error(output_df.loc[index], test_df.loc[index])) 
    WMAPE_step[index] = wmape(output_df.loc[index], test_df.loc[index])
    total_RMSE += np.sum(np.square(output_df.loc[index]- test_df.loc[index]))
    a += np.sum(np.abs(output_df.loc[index]- test_df.loc[index]))
    b += np.sum(np.abs(test_df.loc[index]))

total_RMSE = np.sqrt(total_RMSE/(len(test_df.index)*test_df.shape[1]))
total_WMAPE = a/b

# COMMAND ----------

# MAGIC %md
# MAGIC ###Train a model per group_id

# COMMAND ----------

groups = df_country.sis_group_id.unique()
e_rmse_group = {}
e_wmape_group = {}

for group in groups:
    #Select the rows relative to a certain group_id
    df_group = df_country[df_country['sis_group_id'] == group]
    df_group = df_group.sort_values(['First_day'])
    df_group = df_group[['First_day', 'product_id' , 'total_qty']]
    df_group = df_group.pivot(index = 'First_day', columns= 'product_id', values = 'total_qty')
    scaler = StandardScaler(with_mean=True, with_std=True)
    #Scale the data
    scaler.fit(df_group)
    df_scaled = scaler.transform(df_group)
    df_scaled = pd.DataFrame(df_scaled, index=df_group.index, columns=df_group.columns)
    df_pivot = df_scaled 
    #Create the input and output features of the model 
    x, y = multipleStepSampler(df_pivot, 50)
    SPLIT = 0.85
    X_train = x[:int(SPLIT * len(x))]
    y_train = y[:int(SPLIT * len(y))]
    X_test = x[int(SPLIT * len(x)):]
    y_test = y[int(SPLIT * len(y)):]
    #Train the model
    multivariate_lstm = keras.Sequential()
    multivariate_lstm.add(keras.layers.LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2])))    #model.add(InputLayer((n_steps, n_features)))
    multivariate_lstm.add(keras.layers.Dropout(0.2))
    multivariate_lstm.add(keras.layers.Dense(6 * y_train.shape[2], activation='relu'))  # 12 time steps * n_features( = n_products) per time step
    multivariate_lstm.add(keras.layers.Reshape((6, y_train.shape[2])))  # Reshape output to match y_train shape
    multivariate_lstm.compile(loss = 'MeanSquaredError', metrics=['MAE'], optimizer='Adam')
    history = multivariate_lstm.fit(X_train, y_train, epochs= 10, batch_size = 20)
    #Evaluate the model
    predicted_values = multivariate_lstm.predict(X_test[-1:])
    predicted_values_2d = predicted_values.reshape(-1, predicted_values.shape[-1])
    predicted_values_inversed = scaler.inverse_transform(predicted_values_2d)
    predicted_values_inversed = predicted_values_inversed.reshape(predicted_values.shape)
    y_test_2d = y_test[-1:].reshape(-1, y_test[-1:].shape[-1])
    y_test_inversed = scaler.inverse_transform(y_test_2d)
    y_test_inversed = y_test_inversed.reshape(y_test[-1:].shape)
        # Select the specific indices
    indices_part1 = df_pivot.index[-26:][0:4]  # First 4 indices
    indices_part2 = [df_pivot.index[-26:][11]]  # 12th index, needs to be in a list to concatenate
    indices_part3 = [df_pivot.index[-26:][25]]  # Last index, needs to be in a list to concatenate
    combined_indices = indices_part1.append(pd.to_datetime(indices_part2 + indices_part3))
    result_index = pd.DatetimeIndex(combined_indices)
        #Transform to df
    output_df = pd.DataFrame(predicted_values_inversed[0], index=result_index, columns=df_pivot.columns)
    test_df = pd.DataFrame(y_test_inversed[0], index=result_index, columns=df_pivot.columns)
        #Measure an RMSE and WMAPE per step
    e_rmse = {}
    e_wmape = {}
    for index in test_df.index:
        e_rmse[index] = rmse(output_df.loc[index], test_df.loc[index])
        e_wmape[index] = wmape(output_df.loc[index], test_df.loc[index])
    e_rmse_group[group] = e_rmse
    e_wmape_group[group] = e_wmape

#Transform the RMSE and WMAPE dictionaries into dataframes
d_rmse = pd.DataFrame(e_rmse_group)
d_wmape = pd.DataFrame(e_wmape_group)
#Add a 'total' column to each of the dataframes
d_rmse['total'] = list(RMSE_step.values())
d_wmape['total'] = list(WMAPE_step.values())

# COMMAND ----------

#Modify the index [step1, step2, ... step26] of d_rmse and d_wmape dataframes
d_rmse = d_rmse.reset_index(drop=True)
d_rmse['step'] = [f'step_{i}' for i in [1, 2, 3, 4, 12, 26]]
d_rmse = d_rmse.set_index('step').reset_index()
d_wmape = d_wmape.reset_index(drop=True)
d_wmape['step'] = [f'step_{i}' for i in [1, 2, 3, 4, 12, 26]]
d_wmape = d_wmape.set_index('step').reset_index()

# COMMAND ----------

#Now let's save the measured errors ( Total and per group_id) in 2 SQL tables , one for WMAPE and one for RMSE, to use them to compare the models' performances later.

spark_df = spark.createDataFrame(d_rmse)
spark_df.write.mode("overwrite").saveAsTable(f"RMSE_LSTM_{country}")

spark_df = spark.createDataFrame(d_wmape)
spark_df.write.mode("overwrite").saveAsTable(f"WMAPE_LSTM_{country}")
