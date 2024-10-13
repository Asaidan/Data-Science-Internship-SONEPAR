# Databricks notebook source
# MAGIC %md
# MAGIC ###VAR model

# COMMAND ----------

# MAGIC %md
# MAGIC **The Vector Auto Regression (VAR)** model is one of the most successful, flexible, and easy to use models for the analysis of multivariate time series. It is a natural extension of the univariate autoregressive (AR) model to dynamic multivariate time series. 
# MAGIC
# MAGIC Vector Autoregression (VAR) is a forecasting algorithm that can be used when two or more time series influence each other, i.e. the relationship between the time series involved is bi-directional. It is considered as an Autoregressive model because, each variable (Time Series) is modeled as a function of the past values, that is the predictors are nothing but the lags (time delayed value) of the series.
# MAGIC
# MAGIC Other Autoregressive models like AR, ARMA or ARIMA are uni-directional, where, the predictors influence the Y( unique predicted value) and not vice-versa. Vector Auto Regression (VAR) models are bi-directional, i.e. the variables influence each other.
# MAGIC
# MAGIC In autoregression models, the time series is modeled as a linear combination of it’s own lags. That is, the past values of the series are used to forecast the current and future.
# MAGIC
# MAGIC **A typical **AR(p)** model equation looks something like this :**

# COMMAND ----------

from IPython.display import Image, display
image_path = '/Workspace/Users/anouar.saidan@sonepar.com/anouar.saidan@sonepar.com/Screenshots/AR(p).png'
display(Image(filename=image_path, width = 600))

# COMMAND ----------

# MAGIC
# MAGIC %md
# MAGIC where α is the intercept, a constant and β1, β2 till βp are the coefficients of the lags of Y till order p.
# MAGIC
# MAGIC Order ‘p’ means, up to p-lags of Y is used and they are the predictors in the equation. The ε_{t} is the error, which is considered as white noise.
# MAGIC
# MAGIC In the VAR model, each variable is modeled as a linear combination of past values of itself and the past values of other variables in the system. Since you have multiple time series that influence each other, it is modeled as a system of equations with one equation per variable (time series).
# MAGIC
# MAGIC Let’s suppose, you have two variables (Time series) Y1 and Y2, and you need to forecast the values of these variables at time (t).
# MAGIC
# MAGIC To calculate Y1(t), VAR will use the past values of both Y1 as well as Y2. Likewise, to compute Y2(t), the past values of both Y1 and Y2 be used.
# MAGIC
# MAGIC For example, the system of equations for a **VAR(1) model with two time series (variables `Y1` and `Y2`) is as follows**:

# COMMAND ----------

from IPython.display import Image, display
image_path = '/Workspace/Users/anouar.saidan@sonepar.com/anouar.saidan@sonepar.com/Screenshots/VAR(p).png'
display(Image(filename=image_path, width = 500))

# COMMAND ----------

# MAGIC %md
# MAGIC where, Y{1,t-1} and Y{2,t-1} are the first lag of time series Y1 and Y2 respectively.
# MAGIC
# MAGIC The above equation is referred to as a VAR(1) model, because, each equation is of order 1, that is, it contains up to one lag of each of the predictors (Y1 and Y2).
# MAGIC
# MAGIC Now, let's build a VAR model for our 12165 products (timeseries) dataset.

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
# MAGIC ###Load and transform the data

# COMMAND ----------

from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error

#Evaluation metrics functions
def wmape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))
def rmse(y_true, y_pred):
    res = np.sum((y_true - y_pred)**2)
    n = len(y_true)
    return(np.sqrt(res/n))

#Import data from the prepared SQL table DATA_{country}_Filtered
df = spark.sql(f"""
SELECT *
FROM DATA_{country}_Filtered                   
""")
df_country = df.toPandas()
df_country = df_country.sort_values(['First_day', 'product_id'])

#Transform the format of the data
df = df_country[['First_day', 'product_id' , 'total_qty']]
df_pivot = df.pivot(index='First_day', columns='product_id', values='total_qty')

#Scaling : this type of scaling is called standardization. The resulting values will have a standard deviation of 1 and a mean of 0.
scaler = StandardScaler(with_mean=True, with_std=True)
scaler = scaler.fit(df_pivot)
df_s = scaler.transform(df_pivot)
df_scaled = pd.DataFrame(df_s, index=df_pivot.index, columns=df_pivot.columns)
df_pivot = df_scaled

#Split into train and test datasets 
nobs = 30
df_train, df_test = df_pivot[0:-nobs], df_pivot[-nobs:]

#Check for stationarity : Since the VAR model requires the time series you want to forecast to be stationary, it is customary to check all the time series in the system for stationarity.Let’s use the ADF test for our purpose.
def adfuller_test(series, verbose=False):
    r = adfuller(series, autolag='AIC')
    output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}
    p_value = output['pvalue']
    return(p_value)

seasonal = 0
for column in df_train.columns:
    p = adfuller_test(df_train[column])
    if p > 0.05:
        seasonal +=1

print("number of time series ( = n of products)", len(df_train.columns))
print("number of non-stationary series", seasonal)  

#If a time series are non-stationary, we make it stationary by differencing the series once.
if seasonal> 0:
    df_diff = df_train.diff().dropna()
    products = []
    seasonal = 0
    for column in df_diff.columns:
        p = adfuller_test(df_diff[column])
        if p > 0.05:
            seasonal +=1
            products.append(column)

#The inverse differencing function we will use later          
def invert_transformation(df_train, df_forecast):
    df_fc = df_forecast.copy()
    columns = df_diff.columns
    for col in columns:        
        # Roll back 1st Diff
        df_fc[str(col)+'_forecast'] = df_train[col].iloc[-1] + df_fc[col].cumsum()
    return df_fc
            

# COMMAND ----------

# MAGIC %md
# MAGIC ###Train and evaluate the model

# COMMAND ----------

import warnings
import statsmodels.api as sm

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, message="No frequency information was provided, so inferred frequency")

groups = df_country.sis_group_id.unique()
e_rmse_group = {}
e_wmape_group = {}

for group in groups:

    print(group)
    df_group = df_country[df_country['sis_group_id'] == group]
    df_group = df_group.sort_values(['First_day'])
    df_group = df_group[['First_day', 'product_id' , 'total_qty']]
    df_group = df_group.pivot(index = 'First_day', columns= 'product_id', values = 'total_qty')
    
    #Scaling
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(df_group)
    scaled = scaler.transform(df_group)
    df_scaled = pd.DataFrame(scaled, index=df_group.index, columns=df_group.columns)
    df_pivot = df_scaled

    #Split into train and test
    nobs = 30
    df_train, df_test = df_pivot[0:-nobs], df_pivot[-nobs:]        
    #Treat non-stationary time series 
    df_diff = df_train.diff().dropna()
    #Train the model
    model = VAR(df_diff)        
    result = model.fit(10)
    #Evaluate the model
    forecast = result.forecast(df_diff.values[-result.k_ar:], steps= nobs)
    forecast_df = pd.DataFrame(forecast, index=df_pivot.index[-nobs:], columns=df_diff.columns)
    #Inverse differencing
    df_results = invert_transformation(df_train, forecast_df)   
    columns_to_keep = [col for col in df_results.columns if isinstance(col, str) and col.endswith('forecast')]
    df_results = df_results[columns_to_keep]
    df_results.rename(columns=lambda x: x.replace('_forecast', ''), inplace=True) 
    df_results.columns = df_results.columns.astype('int64')
        #Inverse scaling
    df_results_inversed = scaler.inverse_transform(df_results)
    result = pd.DataFrame(df_results_inversed, index=df_pivot.index[-nobs:], columns=df_results.columns)
    test = scaler.inverse_transform(df_test)
    test_df = pd.DataFrame(test, index=df_pivot.index[-nobs:], columns=df_diff.columns)

    #evaluate the model
    steps = [0,1,3,4,11,25]
    e_rmse= {}
    e_wmape = {}
    for step in steps: 
        e_rmse[step] = rmse(test_df.iloc[step], result.iloc[step]) 
        e_wmape[step] = wmape(test_df.iloc[step], result.iloc[step])
    e_rmse_group[group] = e_rmse
    e_wmape_group[group]= e_wmape

# COMMAND ----------

#Transform dictionaries into dataframes
RM = pd.DataFrame(e_rmse_group)
WM = pd.DataFrame(e_wmape_group)

#Modify the index [step1, step2, ... step26]
RM = RM.reset_index(drop=True)
RM['step'] = [f'step_{i}' for i in [1, 2, 3, 4, 12, 26]]
RM = RM.set_index('step').reset_index()
WM = WM.reset_index(drop=True)
WM['step'] = [f'step_{i}' for i in [1, 2, 3, 4, 12, 26]]
WM = WM.set_index('step').reset_index()

#Save the resulting RMSE and WMAPE values in 2 SQL tables
spark_df = spark.createDataFrame(RM)
spark_df.write.mode("overwrite").saveAsTable(f"RMSE_VAR_{country}")

spark_df = spark.createDataFrame(WM)
spark_df.write.mode("overwrite").saveAsTable(f"WMAPE_VAR_{country}")