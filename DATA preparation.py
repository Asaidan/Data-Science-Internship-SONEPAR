# Databricks notebook source
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


# COMMAND ----------

# MAGIC %md
# MAGIC ###Spark to pandas

# COMMAND ----------

import pandas as pd
import datetime
import pandas as pd
from datetime import timedelta, datetime

df = spark.sql(f"""
SELECT * 
FROM DATA_{country}
""")
df = df.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ###Data preparation

# COMMAND ----------

#Cols is the list of columns used in Data_AUT
cols = ['product_id','total_qty', ...]
df = df[cols]

#Check if there are some null values in the different columns
res = {}
for col in cols:         
    res[col] =  len(df[df[col].isnull()]) == 0
null_cols = [cle for cle, valeur in res.items() if valeur == False]
print("These columns contain some null values", null_cols)
nprods = len(df.product_id.unique())
nprods_null = len(df[df[null_cols].isnull().all(axis=1)].product_id.unique())
print("The percentage of products with null values in those columns is", 100*nprods_null/nprods, "%")

#Drop the rows with null values in null_cols
df = df.dropna(subset=null_cols)
res = {}
for col in cols:         
    res[col] =  len(df[df[col].isnull()]) == 0
null_cols = [cle for cle, valeur in res.items() if valeur == False]
if len(null_cols) == 0:
       print("Now we don't have any null values")

#Adjust the type of numeric features
df['total_qty'] = pd.to_numeric(df['total_qty'])
df['avg_sales_price'] = pd.to_numeric(df['avg_sales_price'])
df['total_discount_value'] = pd.to_numeric(df['total_discount_value'])

#Create the columns 'First_day' ( First_day of the sales_week)
def premier_jour_semaine_annee(numero_semaine, annee):
    premier_janvier = datetime(annee, 1, 1)
    jour_semaine_1er_janvier = premier_janvier.weekday()
    decalage = (numero_semaine - 1) * 7 - jour_semaine_1er_janvier
    premier_jour_semaine = premier_janvier + timedelta(days=decalage)
    premier_jour_semaine =pd.to_datetime(premier_jour_semaine)
    return premier_jour_semaine
df['First_day'] = df.apply(lambda row: premier_jour_semaine_annee(row['sales_week'], row['sales_year']), axis=1)

#Check if there are some missing dates
df = df.sort_values(by='First_day', ascending=False)
date_diffs = pd.Series(df.First_day.unique()).diff(-1)
result = [date_diff == pd.Timedelta(days= 7) for date_diff in date_diffs]
result.pop()
print('There is', result.count(False), 'missing date')  
if (result[1] == False) or (result[-1] == False):
    #Check if it is a min of max date
    if result[-1] == False:
        df_prepared = df[df['First_day'] != df.First_day.unique()[-1]]
    elif result[1] == False:   
        df_prepared = df[df['First_day'] != df.First_day.unique()[1]]

else:
    false_indices = [index for index, value in enumerate(result) if not value]
    missing_dates = []
    def filldate(date, previous_date, next_date, df_oneprod):
        row_ref = df_oneprod[df_oneprod['First_day'] == previous_date]
        row_ref['total_qty']= (df_oneprod[df_oneprod['First_day']== previous_date]['total_qty'].iloc[0] + df_oneprod[df_oneprod['First_day']== next_date]['total_qty'].iloc[0])/ 2
        row_ref['avg_sales_price']= (df_oneprod[df_oneprod['First_day']== previous_date]['avg_sales_price'].iloc[0] + df_oneprod[df_oneprod['First_day']== next_date]['avg_sales_price'].iloc[0]) / 2
        row_ref['total_discount_value']= (df_oneprod[df_oneprod['First_day']== previous_date]['total_discount_value'].iloc[0] +df_oneprod[df_oneprod['First_day']== next_date]['total_discount_value'].iloc[0]) / 2
        row_ref['First_day'] = date
        row_ref['sales_week']  = date.isocalendar()[1]
        row_ref['sales_year'] = date.year
        row_ref['sales_date_weekly_number']= round((df_oneprod[df_oneprod['First_day']== previous_date]['sales_date_weekly_number'].iloc[0] +df_oneprod[df_oneprod['First_day']== next_date]['sales_date_weekly_number'].iloc[0]) / 2)
        return row_ref   
    
    for i in range(len(false_indices)): 
        missing_dates.append(max(pd.Series(df.First_day.unique())) - timedelta(days =(false_indices[i] + 1)*7 ))
    for date in missing_dates :
        df_prepared_copy = df_prepared.copy()
        imputation = pd.DataFrame(columns = df_prepared.columns)
        date_min = df_prepared['First_day'].min()
        date_max = df_prepared['First_day'].max()
        print("missing date", date)
        product_ids = df_prepared_copy['product_id'].unique()

        for product_id in product_ids :
            df_oneprod = df_prepared_copy[df_prepared_copy['product_id']== product_id]
            total_weeks_of_sale = len(df_oneprod['First_day'].unique())
            oneprod_dates = df_oneprod['First_day'].unique()

            if total_weeks_of_sale >= 100:
                previous_date = date - timedelta(days=7)
                next_date = date + timedelta(days =7)

                while (previous_date not in oneprod_dates) & (previous_date >= date_min) :
                    previous_date = previous_date - timedelta(days=7) 

                while (next_date not in oneprod_dates) & (next_date <= date_max):
                    next_date = next_date + timedelta(days=7) 

                if (previous_date in oneprod_dates) & (next_date in oneprod_dates):
                    row =filldate(date, previous_date, next_date, df_oneprod)     
                    imputation = imputation.append(row)

                elif (previous_date in oneprod_dates) & (next_date not in oneprod_dates):
                    row =filldate(date, previous_date, previous_date, df_oneprod)     
                    imputation = imputation.append(row)

                elif (next_date in oneprod_dates) & (previous_date not in oneprod_dates):
                    row =filldate(date, next_date, next_date, df_oneprod)     
                    imputation = imputation.append(row)

            else:
                row_ref = df_oneprod.head(1).copy()
                row_ref['total_qty']= 0
                row_ref['avg_sales_price']= 0
                row_ref['total_discount_value']= 0
                row_ref['First_day'] = date
                row_ref['sales_week']  = date.isocalendar()[1]
                row_ref['sales_year'] = date.year
                row_ref['sales_date_weekly_number']= 0
                row = row_ref
                imputation = imputation.append(row)   

    df_prepared = pd.concat([df_prepared_copy, imputation])

#2nd check after we treated missing dates
date_diffs = pd.Series(df_prepared.First_day.unique()).diff(-1)
result = [date_diff == pd.Timedelta(days= 7) for date_diff in date_diffs]
result.pop()
print('Now, there is', result.count(False), 'missing date') 

#Check if there are some duplicate dates in a single product time series (duplicates in First_day for the same product (end-year weeks))
grouped = df_prepared.groupby(['product_id', 'First_day']).size()
filtered_groups = grouped[grouped > 1]
res = filtered_groups.reset_index().shape[0]
if res > 0:
    print(f'There are {res} duplicate dates')
else: 
    print('There are no duplicate dates')   

#Treate those duplicates 
non_agg_columns = [col for col in df_prepared.columns if col not in ['total_qty', 'total_discount_value', 'avg_sales_price', 'sales_date_weekly_number']]
product_features = df_prepared.drop_duplicates(subset=['product_id', 'First_day'])[non_agg_columns]
grouped = df_prepared.groupby(['product_id', 'First_day'], as_index=False).agg({
    'total_qty': 'sum',
    'total_discount_value': 'sum',
    'avg_sales_price': 'mean',
    'sales_date_weekly_number': 'sum'
})
df_merged = pd.merge(grouped, product_features, on=['product_id', 'First_day'], how='left')
df_prepared = df_merged.reset_index(drop=True)  

#Check after duplicates treatement
grouped = df_prepared.groupby(['product_id', 'First_day']).size()
filtered_groups = grouped[grouped > 1]
res = filtered_groups.reset_index().shape[0]
if res > 0:
    print(f'There are {res} duplicate dates')
else: 
    print('There are no duplicate dates') 

#Take the three last years of dates
df_prepared = df_prepared.sort_values(by='First_day', ascending=False)
dates = df_prepared.First_day.unique()[:158]
df_prepared = df_prepared[df_prepared['First_day'].isin(dates)]

#Create the feature 'Forte_rotation'  
df_prepared['unique_days_count'] = df_prepared.groupby('product_id')['First_day'].transform('nunique')
df_prepared['Forte_rotation'] = df_prepared['unique_days_count'] > 100
df_prepared.drop(columns='unique_days_count', inplace=True)

# COMMAND ----------

#Delete products having less than min_rotation data points
print('Number of Faible rotation products', len(df_prepared[df_prepared['Forte_rotation'] == False].product_id.unique()))
print('Number of Forte rotation products', len(df_prepared[df_prepared['Forte_rotation'] == True].product_id.unique()))
product_counts = df_prepared['product_id'].value_counts()
products_to_delete = product_counts[product_counts <= min_rotation].count()
print(f'Number of products with less than {min_rotation} data point', products_to_delete)

#Delete rows corresponding to products having less than min_rotation
product_counts = df_prepared['product_id'].value_counts()
products_to_keep = product_counts[product_counts > min_rotation].index
df_prepared = df_prepared[df_prepared['product_id'].isin(products_to_keep)]
print('Now, the number of Faible rotation products is', len(df_prepared[df_prepared['Forte_rotation'] == False].product_id.unique()))

# COMMAND ----------

# MAGIC %md
# MAGIC ###Imputation of missing dates for 'Faible_rotation' products

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE DATA_C;
# MAGIC DROP TABLE DATA_NC;
# MAGIC DROP TABLE DATA_UNION;
# MAGIC DROP TABLE DATA_prep;
# MAGIC DROP TABLE DATA_Faible;

# COMMAND ----------

#Create the table DATA_NC
df_prep = df_prepared.drop(columns=['sales_week','sales_year', 'is_stock', 'etim_class', 'etim_class_id', 'sis_sub_family_id', 'sis_sub_family','sis_group_id', 'vendor_id'])
spark_df = spark.createDataFrame(df_prep)
spark_df.write.mode("overwrite").saveAsTable("DATA_NC")

# COMMAND ----------

# MAGIC %sql
# MAGIC --ADD Table_name to DATA_NC
# MAGIC ALTER TABLE DATA_NC
# MAGIC ADD COLUMN Table_name STRING;
# MAGIC UPDATE DATA_NC
# MAGIC SET Table_name = 'DATA_NC';
# MAGIC
# MAGIC --Create DATA_C
# MAGIC --DATA_C has only First_day, product_id and the features we want to complete, as columns.
# MAGIC CREATE OR REPLACE TABLE DATA_C AS
# MAGIC SELECT dates.First_day, products.product_id
# MAGIC FROM (SELECT DISTINCT First_day FROM DATA_NC) dates
# MAGIC CROSS JOIN (SELECT DISTINCT product_id FROM DATA_NC WHERE Forte_rotation == false) products;
# MAGIC --Add columns
# MAGIC ALTER TABLE DATA_C
# MAGIC ADD COLUMN total_qty double;
# MAGIC ALTER TABLE DATA_C
# MAGIC ADD COLUMN avg_sales_price double;
# MAGIC ALTER TABLE DATA_C
# MAGIC ADD COLUMN total_discount_value double;
# MAGIC ALTER TABLE DATA_C
# MAGIC ADD COLUMN Table_name string;
# MAGIC ALTER TABLE DATA_C
# MAGIC ADD COLUMN  sales_date_weekly_number bigint;
# MAGIC --Set zeros in created columns
# MAGIC UPDATE DATA_C
# MAGIC SET Table_name = 'DATA_C', 
# MAGIC total_qty = 0,
# MAGIC avg_sales_price = 0,
# MAGIC total_discount_value=0,
# MAGIC sales_date_weekly_number=0;
# MAGIC
# MAGIC --Count the number of distinct products in DATA_C
# MAGIC SELECT COUNT (DISTINCT product_id) AS prod_C
# MAGIC FROM DATA_C;
# MAGIC
# MAGIC --Drop high turnover products from DATA_NC. Then, count the number of distinct products in DATA_NC
# MAGIC DELETE FROM DATA_NC
# MAGIC WHERE Forte_rotation = TRUE;
# MAGIC SELECT COUNT (DISTINCT product_id)  As prod_NC
# MAGIC FROM DATA_NC;
# MAGIC
# MAGIC -- Create the union table between DATA_C and DATA_NC
# MAGIC CREATE OR REPLACE TABLE DATA_UNION AS
# MAGIC SELECT * FROM DATA_C
# MAGIC UNION ALL
# MAGIC SELECT First_day, product_id, total_qty, avg_sales_price, total_discount_value, Table_name, sales_date_weekly_number FROM DATA_NC;
# MAGIC
# MAGIC --Add the ranking column to the table DATA_UNION and save the resulting table in DATA_prep
# MAGIC CREATE OR REPLACE TABLE DATA_prep AS
# MAGIC SELECT *, RANK() OVER (PARTITION BY First_day, product_id ORDER BY Table_name DESC) AS ranking
# MAGIC FROM DATA_UNION
# MAGIC ORDER BY product_id, First_day;
# MAGIC
# MAGIC --SELECT the lines that correspond to ranking = 1 from DATA_prep and save the result in DATA_faible
# MAGIC CREATE OR REPLACE TABLE DATA_Faible AS
# MAGIC SELECT * FROM DATA_prep
# MAGIC WHERE ranking = 1
# MAGIC

# COMMAND ----------

#SQL table to pandas dataframe
df_faible = spark.sql("""
SELECT * 
FROM DATA_Faible
""")
df_faible = df_faible.toPandas()

# COMMAND ----------

#Add 'sales_week', 'sales_year' and 'Forte_rotation' to df_faible
df_faible['First_day'] = pd.to_datetime(df_faible['First_day'])
df_faible['sales_week'] = df_faible['First_day'].dt.isocalendar().week
df_faible['sales_year'] = df_faible['First_day'].dt.year
df_faible['Forte_rotation'] = False
df_faible = df_faible.drop(columns=['Table_name'])

# COMMAND ----------

if len(df_faible) == len(df_prepared[df_prepared['Forte_rotation']== False].product_id.unique()) * len(df_prepared.First_day.unique()) :
    print('df_faible is a dataframe with complete dates for all low turnover products')
else:
    print('Review your code')   

# COMMAND ----------

# MAGIC %md
# MAGIC ###Imputation of missing dates for 'Forte_rotation' products

# COMMAND ----------

from datetime import timedelta, datetime
import pandas as pd

df_prepared_copy = df_prepared[['product_id', 'sales_week', 'sales_year', 'total_qty', 'avg_sales_price', 'total_discount_value', 'First_day', 'Forte_rotation', 'sales_date_weekly_number']]

date_min = df_prepared_copy['First_day'].min()
date_max = df_prepared_copy['First_day'].max()

df_forte = df_prepared_copy[df_prepared_copy['Forte_rotation'] == True]
product_ids = df_forte['product_id'].unique()

def filldate(date, previous_date, next_date, df_oneprod):
    previous_row = df_oneprod.loc[previous_date]
    next_row = df_oneprod.loc[next_date]
    row = previous_row.copy()
    row['total_qty'] = (previous_row['total_qty'] + next_row['total_qty']) / 2
    row['avg_sales_price'] = (previous_row['avg_sales_price'] + next_row['avg_sales_price']) / 2
    row['total_discount_value'] = (previous_row['total_discount_value'] + next_row['total_discount_value']) / 2
    row['First_day'] = date
    row['sales_week'] = date.isocalendar()[1]
    row['sales_year'] = date.year
    row['sales_date_weekly_number'] = round((previous_row['sales_date_weekly_number'] + next_row['sales_date_weekly_number']) / 2)
    return row

rows_to_add = []
for product_id in product_ids:
    df_oneprod = df_forte[df_forte['product_id'] == product_id].set_index('First_day')
    oneprod_dates = df_oneprod.index
    all_dates = pd.date_range(start=date_min, end=date_max, freq='7D')
    missing_dates = all_dates.difference(oneprod_dates)
    for date in missing_dates:
        previous_date = date - timedelta(days=7)
        next_date = date + timedelta(days=7)
        while previous_date not in oneprod_dates and previous_date > date_min:
            previous_date = previous_date - timedelta(days=7)
        while next_date not in oneprod_dates and next_date < date_max:
            next_date = next_date + timedelta(days=7)
        if previous_date in oneprod_dates and next_date in oneprod_dates:
            row = filldate(date, previous_date, next_date, df_oneprod)
            rows_to_add.append(row)
        elif previous_date in oneprod_dates and next_date not in oneprod_dates:
            row = filldate(date, previous_date, previous_date, df_oneprod)
            rows_to_add.append(row)
        elif next_date in oneprod_dates and previous_date not in oneprod_dates:
            row = filldate(date, next_date, next_date, df_oneprod)
            rows_to_add.append(row)

if rows_to_add:
    imputation = pd.DataFrame(rows_to_add)


# COMMAND ----------

#Concatenate imputation and df_forte_NC to have a dataframe with complete dates for all high turnover products
df_prepared_copy = pd.DataFrame(df_prepared[['product_id', 'sales_week', 'sales_year','total_qty','avg_sales_price', 'total_discount_value', 'First_day', 'Forte_rotation', 'sales_date_weekly_number']])
df_forte_NC = df_prepared_copy[df_prepared_copy['Forte_rotation']== True]
df_forte = pd.concat([df_forte_NC, imputation])
if len(df_forte) == len(df_prepared[df_prepared['Forte_rotation']].product_id.unique()) * len(df_prepared.First_day.unique()) :
    print('df_forte is a dataframe with complete dates for all high turnover products')
else:
    print('Review your code')    

# COMMAND ----------

df_forte_s = df_forte.sort_values(by='First_day', ascending=False)
df_faible_s = df_faible.sort_values(by='First_day', ascending=False)

# COMMAND ----------

#Concatenate df_faible and df_forte in one dataframe df_complete
df_complete = pd.concat([df_faible, df_forte])

#Add the missing columns to df_complete from df_prepared
cols = ['is_stock', 'etim_class', 'etim_class_id', 'sis_sub_family', 'sis_sub_family_id', 'sis_group_id', 'vendor_id']
df_unique = df_prepared[['product_id'] + cols].drop_duplicates(subset='product_id')
df_complete = df_complete.merge(df_unique, on='product_id', how='left')

#Drop 'ranking' column from df_complete
df_complete = df_complete.drop(columns = ['ranking'])

#save df_complete, which now has complete rows and complete columns, in the SQL table DATA_{country}_Final
spark_df = spark.createDataFrame(df_complete)
spark_df.write.mode("overwrite").saveAsTable(f"DATA_{country}_FINAL")

# COMMAND ----------

# MAGIC %md
# MAGIC ###EDA to detect outliers

# COMMAND ----------

df = spark.sql(f"""
SELECT * 
FROM DATA_{country}_FINAL
""")
df_complete = df.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ###EDA to detect outliers : Total_qty

# COMMAND ----------

#Imputation of outliers :  total_qty
#As we want to keep all the rows in the dataset, instead of deleting the rows showing outliers in total_qty, we will put the average of the values from the following date and the preceding date
from datetime import timedelta, datetime
import pandas as pd
import numpy as np

#Define a threshold beyond which a total_qty value is considered as outlier
mean = np.mean(df_prepared.total_qty)
std_dev = np.std(df_prepared.total_qty)
lower_value = mean - 3*std_dev
higher_value = mean + 3*std_dev

#df_outliers contains the rows of DATA_complete that have outliers in total_qty
df_outliers = df_complete[df_complete['total_qty'] >= higher_value]
product_ids= df_outliers.product_id.unique()
product_ids_list =product_ids.tolist()

#For each product, we define the dates having outliers in total_qty and the dates having normal values in total_qty. For each date having outlier in total_qty, we create a row that contains the average of the values from the preceding date and the followig date in total qty. Then, we add this row to the dataframe 'imputation'
rows_to_add = []
for product_id in product_ids:
    df_oneprod = df_complete[df_complete['product_id']== product_id]
    dates_out = df_outliers[df_outliers['product_id']== product_id].First_day.unique()
    dates_out_series =  pd.Series(dates_out)
    df_oneprod_filtered = df_oneprod[df_oneprod['total_qty']< higher_value]
    dates_filtered = df_oneprod_filtered.First_day.unique()
    dates_filtered = sorted(dates_filtered)

    for date in dates_out_series:
        row = df_oneprod[df_oneprod['First_day']== date]
        date_np = np.datetime64(date)
        index = np.searchsorted(dates_filtered, date_np)

        if (index > 0) & (index < len(dates_filtered)):
            previous_date = dates_filtered[index -1]
            next_date = dates_filtered[index]
            previous_val = df_oneprod[df_oneprod['First_day']== previous_date]['total_qty'].iloc[0]
            next_val = df_oneprod[df_oneprod['First_day']== next_date]['total_qty'].iloc[0]
            row['total_qty'] =  (next_val + previous_val)/2
            
        elif index == 0:
            next_date = dates_filtered[index]
            next_val = df_oneprod[df_oneprod['First_day']== next_date]['total_qty'].iloc[0]
            row['total_qty'] =  next_val

        elif index == len(dates_filtered):
            previous_date = dates_filtered[index -1]
            previous_val = df_oneprod[df_oneprod['First_day']== previous_date]['total_qty'].iloc[0]
            row['total_qty'] =  previous_val

        rows_to_add.append(row)

imputation = pd.concat(rows_to_add, ignore_index=True)
###To make to code run faster : Use the search sorted function and et use the imputation dataframe instead of adding directly on df_complete

# COMMAND ----------

df_tq = pd.concat([df_complete, imputation])
df_tq = df_tq[df_tq['total_qty']< higher_value] 
df_complete = df_tq

# COMMAND ----------

# MAGIC %md
# MAGIC ###EDA to detect outliers : Avg_sales_price

# COMMAND ----------

#Imputation of outliers :  Avg_sales_price
#As we want to keep all the rows in the dataset, instead of deleting the rows showing outliers in avg_sales_price, we will put the average of the values from the following date and the preceding date

mean = np.mean(df_prepared.avg_sales_price)
std_dev = np.std(df_prepared.avg_sales_price)
lower_value = mean - 3*std_dev
higher_value = mean + 3*std_dev


df_outliers = df_complete[df_complete['avg_sales_price'] >= higher_value]
product_ids= df_outliers.product_id.unique()
product_ids_list =product_ids.tolist()

rows_to_add = []
for product_id in product_ids:
    df_oneprod = df_complete[df_complete['product_id']== product_id]
    dates_out = df_outliers[df_outliers['product_id']== product_id].First_day.unique()
    dates_out_series =  pd.Series(dates_out)
    df_oneprod_filtered = df_oneprod[df_oneprod['avg_sales_price']< higher_value]
    dates_filtered = df_oneprod_filtered.First_day.unique()
    dates_filtered = sorted(dates_filtered)

    for date in dates_out_series:
        row = df_oneprod[df_oneprod['First_day']== date]
        date_np = np.datetime64(date)
        index = np.searchsorted(dates_filtered, date_np)

        if (index > 0) & (index < len(dates_filtered)):
            previous_date = dates_filtered[index -1]
            next_date = dates_filtered[index]
            previous_val = df_oneprod[df_oneprod['First_day']== previous_date]['avg_sales_price'].iloc[0]
            next_val = df_oneprod[df_oneprod['First_day']== next_date]['avg_sales_price'].iloc[0]
            row['avg_sales_price'] =  (next_val + previous_val)/2
            
        elif index == 0:
            next_date = dates_filtered[index]
            next_val = df_oneprod[df_oneprod['First_day']== next_date]['avg_sales_price'].iloc[0]
            row['avg_sales_price'] =  next_val

        elif index == len(dates_filtered):
            previous_date = dates_filtered[index -1]
            previous_val = df_oneprod[df_oneprod['First_day']== previous_date]['avg_sales_price'].iloc[0]
            row['avg_sales_price'] =  previous_val

        rows_to_add.append(row)

imputation = pd.concat(rows_to_add, ignore_index=True)
###To make to code run faster : Use the search sorted function and et use the imputation dataframe instead of adding directly on df_complete

# COMMAND ----------

df_avg = pd.concat([df_complete, imputation])
df_avg = df_avg[df_avg['avg_sales_price']< higher_value] 
df_complete = df_avg

# COMMAND ----------

# MAGIC %md
# MAGIC ###EDA to detect outliers : Total_discount_value

# COMMAND ----------

#Imputation of outliers :  Total_discount_value
#As we want to keep all the rows in the dataset, instead of deleting the rows showing outliers in total_discount_value, we will put the average of the values from the following date and the preceding date

mean = np.mean(df_prepared.total_discount_value)
std_dev = np.std(df_prepared.total_discount_value)
lower_value = mean - 3*std_dev
higher_value = mean + 3*std_dev

df_outliers = df_complete[df_complete['total_discount_value'] >= higher_value]
product_ids= df_outliers.product_id.unique()
product_ids_list =product_ids.tolist()

rows_to_add = []
for product_id in product_ids:
    df_oneprod = df_complete[df_complete['product_id']== product_id]
    dates_out = df_outliers[df_outliers['product_id']== product_id].First_day.unique()
    dates_out_series =  pd.Series(dates_out)
    df_oneprod_filtered = df_oneprod[df_oneprod['total_discount_value']< higher_value]
    dates_filtered = df_oneprod_filtered.First_day.unique()
    dates_filtered = sorted(dates_filtered)

    for date in dates_out_series:
        row = df_oneprod[df_oneprod['First_day']== date]
        date_np = np.datetime64(date)
        index = np.searchsorted(dates_filtered, date_np)

        if (index > 0) & (index < len(dates_filtered)):
            previous_date = dates_filtered[index -1]
            next_date = dates_filtered[index]
            previous_val = df_oneprod[df_oneprod['First_day']== previous_date]['total_discount_value'].iloc[0]
            next_val = df_oneprod[df_oneprod['First_day']== next_date]['total_discount_value'].iloc[0]
            row['total_discount_value'] =  (next_val + previous_val)/2
            
        elif index == 0:
            next_date = dates_filtered[index]
            next_val = df_oneprod[df_oneprod['First_day']== next_date]['total_discount_value'].iloc[0]
            row['total_discount_value'] =  next_val

        elif index == len(dates_filtered):
            previous_date = dates_filtered[index -1]
            previous_val = df_oneprod[df_oneprod['First_day']== previous_date]['total_discount_value'].iloc[0]
            row['total_discount_value'] =  previous_val

        rows_to_add.append(row)

imputation = pd.concat(rows_to_add, ignore_index=True)
###To make to code run faster : Use the search sorted function and et use the imputation dataframe instead of adding directly on df_complete

# COMMAND ----------

df_disc = pd.concat([df_complete, imputation])
df_disc = df_disc[df_disc['avg_sales_price']< higher_value] 
df_complete = df_disc

# COMMAND ----------

#Save the filtered dataframe in a SQL table DATA_{country}_Filtered
spark_df = spark.createDataFrame(df_complete)
spark_df.write.mode("overwrite").saveAsTable(f"DATA_{country}_Filtered")
