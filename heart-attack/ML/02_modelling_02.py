# Databricks notebook source
import pandas as pd

# COMMAND ----------

#%sql
#SELECT * FROM `hive_metastore`.`default`.`heart`;
df_Spark = spark.read.table("hive_metastore.default.heart")
df = df_Spark.toPandas()



# COMMAND ----------

from pyspark.sql import functions as F
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Modeling

# COMMAND ----------


