# Databricks notebook source
import pandas as pd

# COMMAND ----------

#%sql
#SELECT * FROM `hive_metastore`.`default`.`heart`;
df_Spark = spark.read.table("hive_metastore.default.heart_attack_gold")
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


import mlflow
import mlflow.lightgbm
import lightgbm as lgb
import shap
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

# COMMAND ----------

# MAGIC %md
# MAGIC ##Modeling

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Variáveis relevantes: sex, age, cp, thall, oldpeak, exng, restecg, caa

# COMMAND ----------

y = df["output"]
X = df.drop(columns="output")
X = df.drop(columns="trtbps")
X = df.drop(columns="chol")
X = df.drop(columns="fbs")

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# COMMAND ----------

mlflow.start_run()

# COMMAND ----------

#cat_cols = ['sex','exng','caa','cp','fbs','restecg','slp','thall']
#con_cols = ["age","trtbps","chol","thalachh","oldpeak"]

cat_cols = ['sex','exng','caa','cp','restecg','slp','thall']
con_cols = ["age","thalachh","oldpeak"]

target_col = ["output"]
print("The categorial cols are : ", cat_cols)
print("The continuous cols are : ", con_cols)
print("The target variable is :  ", target_col)

# COMMAND ----------

from sklearn.preprocessing import RobustScaler

# Train Test Split
from sklearn.model_selection import train_test_split

# Models
import torch
import torch.nn as nn
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Metrics
from sklearn.metrics import accuracy_score, classification_report, roc_curve

# Cross Validation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

print('Packages imported...')

# COMMAND ----------

# creating a copy of df
df1 = df

# define the columns to be encoded and scaled
#cat_cols = ['sex','exng','caa','cp','fbs','restecg','slp','thall']
#con_cols = ["age","trtbps","chol","thalachh","oldpeak"]

cat_cols = ['sex','exng','caa','cp','restecg','slp','thall']
con_cols = ["age","thalachh","oldpeak"]

target_col = ["output"]
print("The categorial cols are : ", cat_cols)
print("The continuous cols are : ", con_cols)
print("The target variable is :  ", target_col)

# encoding the categorical columns
df1 = pd.get_dummies(df1, columns = cat_cols, drop_first = True)

# defining the features and target
X = df1.drop(['output'],axis=1)
X = df1.drop(columns="output")
X = df1.drop(columns="trtbps")
X = df1.drop(columns="chol")
X = df1.drop(columns="fbs")

y = df1[['output']]

# instantiating the scaler
scaler = RobustScaler()

# scaling the continuous featuree
X[con_cols] = scaler.fit_transform(X[con_cols])
print("The first 5 rows of X are")
X.head()

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)
print("The shape of X_train is      ", X_train.shape)
print("The shape of X_test is       ",X_test.shape)
print("The shape of y_train is      ",y_train.shape)
print("The shape of y_test is       ",y_test.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Linear Classifiers

# COMMAND ----------

# instantiating the object and fitting
clf = SVC(kernel='linear', C=1, random_state=42).fit(X_train,y_train)

# predicting the values
y_pred = clf.predict(X_test)

# printing the test accuracy
print("The test accuracy score of SVM is ", accuracy_score(y_test, y_pred))

# COMMAND ----------

# instantiating the object
svm = SVC()

# setting a grid - not so extensive
parameters = {"C":np.arange(1,10,1),'gamma':[0.00001,0.00005, 0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,1,5]}

# instantiating the GridSearchCV object
searcher = GridSearchCV(svm, parameters)

# fitting the object
searcher.fit(X_train, y_train)

# the scores
print("The best params are :", searcher.best_params_)
print("The best score is   :", searcher.best_score_)

# predicting the values
y_pred = searcher.predict(X_test)

# printing the test accuracy
print("The test accuracy score of SVM after hyper-parameter tuning is ", accuracy_score(y_test, y_pred))

# COMMAND ----------

# MAGIC %md
# MAGIC ###Hyperparameter tuning of SVC
