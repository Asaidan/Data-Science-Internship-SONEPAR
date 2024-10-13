# Databricks notebook source
# MAGIC %md
# MAGIC ###ARIMA

# COMMAND ----------

# MAGIC %md
# MAGIC ARIMA, short for ‘**Auto Regressive Integrated Moving Average**’ is actually a class of models that ‘explains’ a given time series based on its own past values, that is, its own lags and the lagged forecast errors, so that equation can be used to forecast future values.
# MAGIC
# MAGIC An ARIMA model is characterized by 3 terms: p, d, q
# MAGIC
# MAGIC where,
# MAGIC
# MAGIC - p is the order of the AR term:  it refers to the number of lags of Y to be used as predictors
# MAGIC
# MAGIC - q is the order of the MA term: it refers to the number of lagged forecast errors that should go into the ARIMA Model
# MAGIC
# MAGIC - d is the number of differencing required to make the time series stationary

# COMMAND ----------

# MAGIC %md
# MAGIC ###Parameters 

# COMMAND ----------

import configparser

config = configparser.ConfigParser()
files_read = config.read('/Workspace/Users/anouar.saidan@sonepar.com/anouar.saidan@sonepar.com/config.ini')
country = config['parameters']['country']
print(f"Country: {country}")

# COMMAND ----------

# MAGIC %md
# MAGIC ###Load and prepare the data

# COMMAND ----------

! pip install pmdarima

# COMMAND ----------

import warnings
import pandas as pd
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from statsmodels.tsa.stattools import adfuller

#Load the data
df = spark.sql(f"""
SELECT *
FROM DATA_{country}_Filtered                   
""")
df =df.toPandas()

#Sort the dataframe
df = df.sort_values(['First_day', 'product_id'])

#Calculate COV indicateur for each product
COV = {}
product_ids = df.product_id.unique()
for product_id in product_ids:
    df_p = df[df['product_id']== product_id]['total_qty']
    COV[product_id] = df_p.std()/df_p.mean()

#Calculate diff indicator for each product
diff = {}
product_ids = df.product_id.unique()
for product_id in product_ids:
    df_oneprod = df[df['product_id']== product_id]
    result = adfuller(df_oneprod.total_qty)
    if result[1] < 0.05:
        diff[product_id] = 0
    else:
        diff[product_id] = 1      

#Define performance metrics functions
def wmape(ytrue, ypred):
    return np.sum(np.abs(ytrue - ypred)) / np.sum(np.abs(ytrue))
def rmse(y_true, y_pred):
    res = np.sum((y_true - y_pred)**2)
    n = len(y_true)
    return(np.sqrt(res/n))
    

# COMMAND ----------

# MAGIC %md
# MAGIC ###Apply ARIMA to all the products' time series

# COMMAND ----------

#Train ARIMA model with order (0, d, q) for high covariance products

#Select high covariance products from the COV dictionary
COV_high = {product_id: cov for product_id, cov in COV.items() if cov >= 1}
wmape_high_COV = {}
rmse_high_COV = {}
pred_high_COV = {}

for prod in COV_high.keys(): 
    df_prod = df[df['product_id']== prod]
    d = diff[prod]
    X = df_prod.total_qty
    SPLIT = 0.85
    train_size = int(SPLIT * len(X))
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    #Train the model for the selected product
    model = ARIMA(history, order=(0,d,0))
    model_fit = model.fit()
    #Evaluate the model
    yhat = model_fit.forecast(steps = 24)
    pred_high_COV[prod] = yhat
    wmape_high_COV[prod] = wmape(test, yhat)
    rmse_high_COV[prod] = rmse(test, yhat)

# COMMAND ----------

#Train auto_arima model for low covariance products

#Select low covariance products from the COV dictionary
COV_low = {product_id: cov for product_id, cov in COV.items() if cov < 1}
wmape_low_COV = {}
rmse_low_COV = {}
pred_low_COV = {}

for prod in COV_low.keys():
    df = df[df['product_id']== prod] 
    d = diff[prod]
    X = df_prod.total_qty
    SPLIT = 0.85
    train_size = int(SPLIT * len(X))
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    model = auto_arima(y=history, d = d, stepwise=True, start_p = 20, max_p = 100, start_q = 20, max_q = 100, information_criterion='aic',
    trace=True ,seasonal = False)
    yhat = model.predict(n_periods=24)
    pred_low_COV[prod] = yhat
    wmape_low_COV[prod] = wmape(test, yhat)
    rmse_low_COV[prod] = rmse(test, yhat)


# COMMAND ----------

#Total_RMSE and Total_WMAPE for each step

steps = [1, 2, 3, 4, 12, 24]
rmse_step = {}
wmape_step = {}

for step in steps:
    e = 0
    a = 0
    b = 0
    for prod in COV_low.keys():
        pred = pred_low_COV[prod][step-1]
        X = df[df['product_id']== prod].total_qty
        tests = X[int(0.85 * len(X)):]
        test = tests.iloc[step -1]
        e += (pred -  test)**2 
        a += np.abs(pred - test)
        b += np.abs(test)

    for prod in COV_high.keys():
        pred = pred_high_COV[prod][step-1]
        X = df[df['product_id']== prod].total_qty
        tests = X[int(0.85 * len(X)):]
        test = tests.iloc[step -1]
        e += (pred -  test)**2   
        a += np.abs(pred - test)
        b += np.abs(test) 
 
    rmse_step[step] = np.sqrt(e/len(df.product_id.unique()))
    wmape_step[step] = a/b
