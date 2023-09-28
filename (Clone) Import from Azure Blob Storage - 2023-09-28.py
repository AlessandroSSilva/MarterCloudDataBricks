# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC This notebook shows you how to create and query a table or DataFrame loaded from data stored in Azure Blob storage.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Step 1: Set the data location and type
# MAGIC
# MAGIC There are two ways to access Azure Blob storage: account keys and shared access signatures (SAS).
# MAGIC
# MAGIC To get started, we need to set the location and type of the file.

# COMMAND ----------

storage_account_name = "saalphavantagealess"
storage_account_access_key = "/fUh+n8PfAb+Q/qNXiRRQnC+wFNL1naMO+FYD0MI6mAUBo1awu/Sm3BrMyGnBCrfThBbkyhFAt6f+AStr3yjCg=="

# COMMAND ----------

file_location = "wasbs://alphavantage2/bronze"
file_type = "json"

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Step 2: Read the data
# MAGIC
# MAGIC Now that we have specified our file metadata, we can create a DataFrame. Notice that we use an *option* to specify that we want to infer the schema from the file. We can also explicitly set this to a particular schema if we have one already.
# MAGIC
# MAGIC First, let's create a DataFrame in Python.

# COMMAND ----------

dbutils.fs.mount(
source = "wasbs://alphavantage2@saalphavantagealess.blob.core.windows.net/",
mount_point = "/mnt/alphavantage",
extra_configs = {"fs.azure.account.key.saalphavantagealess.blob.core.windows.net": "/fUh+n8PfAb+Q/qNXiRRQnC+wFNL1naMO+FYD0MI6mAUBo1awu/Sm3BrMyGnBCrfThBbkyhFAt6f+AStr3yjCg=="})

# COMMAND ----------

# MAGIC %fs ls /mnt/alphavantage/bronze

# COMMAND ----------

df = spark.read.json("/mnt/alphavantage/bronze")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Step 3: Query the data
# MAGIC
# MAGIC Now that we have created our DataFrame, we can query it. For instance, you can identify particular columns to select and display.

# COMMAND ----------

display(df.select("*"))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Step 4: (Optional) Create a view or table
# MAGIC
# MAGIC If you want to query this data as a table, you can simply register it as a *view* or a table.

# COMMAND ----------

df.createOrReplaceTempView("AlphaVantageView")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC We can query this view using Spark SQL. For instance, we can perform a simple aggregation. Notice how we can use `%sql` to query the view from SQL.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT date_reference, count(*) FROM AlphaVantageView GROUP BY date_reference
# MAGIC --SELECT * FROM AlphaVantageView
# MAGIC --SELECT symbol, date_reference, min(low), max(low) FROM AlphaVantageView  GROUP BY symbol, date_reference

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Since this table is registered as a temp view, it will be available only to this notebook. If you'd like other users to be able to query this table, you can also create a table from the DataFrame.

# COMMAND ----------

df.write.format("parquet").saveAsTable("AlphaVantage")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC This table will persist across cluster restarts and allow various users across different notebooks to query this data.

# COMMAND ----------

from pyspark.sql import functions as F

# COMMAND ----------

df = spark.read.json("/mnt/alphavantage/bronze")
df = df.withColumnRenamed("Time Series (Daily)", "TimeSeries")


# COMMAND ----------

from pyspark.sql.functions import col, explode
df.selectExpr("explode(map_from_entries(named_struct('date', date_reference, 'open'))) as DailyTimeSeries")\
  .select("*")


# COMMAND ----------

df.show()

# COMMAND ----------

display(
  df
  .groupby("date_reference")
  .agg(
    F.count("ticket")
  )
  .write.format("parquet").mode("overwrite").saveAsTable("alphavantage_tickets")
)
