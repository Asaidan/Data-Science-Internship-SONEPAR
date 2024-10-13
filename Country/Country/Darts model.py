# Databricks notebook source
#Install the darts library
!pip install darts

# COMMAND ----------

#Restart the kernel
dbutils.library.restartPython() 

# COMMAND ----------

# MAGIC %md
# MAGIC ###Parameters 

# COMMAND ----------

import configparser

config = configparser.ConfigParser()
files_read = config.read('/Workspace/Users/anouar.saidan@sonepar.com/anouar.saidan@sonepar.com/config.ini')
country = config['parameters']['country']
min_rotation = int(config['parameters']['min_rotation'])
print(f"Country: {country}")

# COMMAND ----------

# MAGIC %md
# MAGIC ###Load and prepare the data

# COMMAND ----------

from darts import TimeSeries
from darts.models import RegressionModel
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import itertools

#Load the data from DATA_{country}_Filtered
df = spark.sql(f"""
SELECT * 
FROM DATA_{country}_Filtered  
""")
df = df.toPandas()

#sort by First_day then product_id
df = df.sort_values(['First_day', 'product_id'])

#Define the performance metrics calculation functions
def wmape(ytrue, ypred):
    return np.sum(np.abs(ytrue - ypred)) / np.sum(np.abs(ytrue))

# Create time series from df after grouping data by 'product_id'
y_all = TimeSeries.from_group_dataframe(df, group_cols = ['product_id'], time_col = 'First_day', value_cols = ['total_qty'])  

#Split the data on train, validation and test datasets
#Validation dataset is used to tune the models hyperparameters
y_all_tv = [ sub[:132] for sub in y_all]
y_all_train = [ sub[:100] for sub in y_all_tv]
y_all_valid = [ sub[100:] for sub in y_all_tv]
y_all_test = [ sub[132:] for sub in y_all]

# COMMAND ----------

# MAGIC %md
# MAGIC ###Choose the best combination of model parameters

# COMMAND ----------

#Specifiy the different parameter values the model can take
lags = [[-1, -2, -3, -4, -12, -52], [-1, -2, -3, -4, -12, -26, -52]]
alphas = [0.1, 0.2, 0.001]
futures = [['sales_year'], ['sales_week'], ['sales_year', 'sales_week']]
fit_intercepts = [True, False]
chunks = [1, 2, 6, 12, 26 ]
param_combi = list(itertools.product(lags, alphas, futures, fit_intercepts, chunks))

#Iterate to find the best combination ( lowest RMSE value)
best_combi= []
best_rmse = 100000
for lag, alpha, future, fit_intercept, chunk in param_combi:
     
    #future_cov are additional features that can help improve the accuracy of the model's predictions
    future_cov = TimeSeries.from_group_dataframe(df, group_cols = ['product_id'], time_col = 'First_day', value_cols = future)
    #Train and fit the darts LinearRegression model , we use Ridge to have positive values as output ( total_qty > 0)
    LRmodel = RegressionModel(lags= lag, model=Ridge(alpha= alpha, fit_intercept= fit_intercept, positive=True), output_chunk_length= chunk, lags_future_covariates=[0])
    LRmodel.fit(y_all_train,  future_covariates= future_cov)
    #Evaluate the model
    y_pred = LRmodel.predict(n = 32,  series = y_all_train, future_covariates= future_cov)
        #Measure the total RMSE value for the current model
    error = 0
    for i in range(len(y_pred)):
        e = mean_squared_error(y_pred[i].values(), y_all_valid[i].values())*len(y_pred[i])
        error += e
    rm = np.sqrt(error/(len(y_pred)*len(y_pred[0])))
    if rm< best_rmse:
        best_rmse = rm
        best_combi = [lag, alpha, future, fit_intercept, chunk]

# COMMAND ----------

best_rmse

# COMMAND ----------

# MAGIC %md
# MAGIC ###Train and evaluate the selected model

# COMMAND ----------

lag, alpha, future, fit_intercept, chunk = best_combi
future_cov = TimeSeries.from_group_dataframe(df, group_cols = ['product_id','Forte_rotation'], time_col = 'First_day', value_cols = future)
LRmodel = RegressionModel(lags= lag, model=Ridge(alpha= alpha, fit_intercept= fit_intercept, positive=True), output_chunk_length= chunk, lags_future_covariates=[0])
LRmodel.fit(y_all_tv,  future_covariates= future_cov)
#Evaluate the model 
y_pred = LRmodel.predict(n = 26,  series = y_all_tv, future_covariates= future_cov)
    #Total_RMSE, Total_wmape per step
steps = [1, 2, 3, 4, 12, 26]
rmse_step ={}
wmape_step ={}
for step in steps :
    pred = [ sub[step -1] for sub in y_pred]
    test = [ sub[step -1] for sub in y_all_test]
    error = 0
    a = 0
    b= 0
    for i in range(len(pred)):
        e = mean_squared_error(pred[i].values(), test[i].values())*len(pred[i])
        error += e
        a += np.abs( pred[i].values()[0][0] -  test[i].values()[0][0] )
        b += np.abs( pred[i].values()[0][0])
    rmse_step[step] = np.sqrt(error/(len(pred)*len(pred[0])))    
    wmape_step[step] = a/b
rmse_group = {}
wmape_group = {}
groups = df.sis_group_id.unique()
for group in groups : 
    prods = df[df['sis_group_id'] == group].product_id.unique()
    y_pred_group = [ y_pred[i] for i in range(len(y_pred)) if  y_pred[i].static_covariates['product_id'].item() in prods]
    y_test_group = [ y_all_test[i] for i in range(len(y_all_test)) if  y_all_test[i].static_covariates['product_id'].item() in prods]
    rm_step = {}
    wm_step = {}
    for step in steps:
        pred = [ sub[step -1] for sub in y_pred_group]
        test = [ sub[step -1] for sub in y_test_group]
        error = 0
        a = 0
        b= 0
        for i in range(len(pred)):
            e = mean_squared_error(pred[i].values(), test[i].values())*len(pred[i])
            error += e
            a += np.abs( pred[i].values()[0][0] -  test[i].values()[0][0] )
            b += np.abs( pred[i].values()[0][0])
        rm_step[step]  =  np.sqrt(error/(len(pred)*len(pred[0]))) 
        wm_step[step] = a/b
    rmse_group[group] = rm_step
    wmape_group[group] = wm_step    

# COMMAND ----------

#Save the RMSE and WMAPE dictionaries into dataframes 
WMAPE = pd.DataFrame(wmape_group)
WMAPE['total'] = wmape_step
RMSE = pd.DataFrame(rmse_group)
RMSE['total'] = rmse_step
#save the dataframes into SQL tables
spark_df = spark.createDataFrame(RMSE)
spark_df.write.mode("overwrite").saveAsTable(f"RMSE_Darts_{country}")
spark_df = spark.createDataFrame(WMAPE)
spark_df.write.mode("overwrite").saveAsTable(f"WMAPE_Darts_{country}")
