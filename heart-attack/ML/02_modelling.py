# Databricks notebook source
import mlflow
import mlflow.lightgbm
import lightgbm as lgb
import shap
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np # linear algebra

from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler


import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.svm import SVC

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

spark_df = spark.read.table("hive_metastore.default.heart_attack_gold")
df = spark_df.toPandas()

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC #Modelagem

# COMMAND ----------

# MAGIC %md
# MAGIC ## Dados de treino e teste

# COMMAND ----------

x = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values
x,y

# COMMAND ----------

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state= 0)

# COMMAND ----------

print('Shape for training data', x_train.shape, y_train.shape)
print('Shape for testing data', x_test.shape, y_test.shape)

# COMMAND ----------

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# COMMAND ----------

x_train,x_test

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC ##Logistic Regression

# COMMAND ----------

from sklearn.linear_model import LogisticRegression

# COMMAND ----------

model = LogisticRegression()
model.fit(x_train, y_train)
predicted=model.predict(x_test)
conf = confusion_matrix(y_test, predicted)
print ("Matriz de Confusão: \n", conf)
print()
print()
print ("The accuracy of Logistic Regression is : ", accuracy_score(y_test, predicted)*100, "%")

# COMMAND ----------

# MAGIC %md
# MAGIC ##Gaussian Naive Bayes
# MAGIC

# COMMAND ----------

model = GaussianNB()
model.fit(x_train, y_train)
  
predicted = model.predict(x_test)
  
print("The accuracy of Gaussian Naive Bayes model is : ", accuracy_score(y_test, predicted)*100, "%")

# COMMAND ----------

# MAGIC %md
# MAGIC ##Bernoulli Naive Bayes

# COMMAND ----------

model = BernoulliNB()
model.fit(x_train, y_train)
  
predicted = model.predict(x_test)
  
print("The accuracy of Gaussian Naive Bayes model is : ", accuracy_score(y_test, predicted)*100, "%")

# COMMAND ----------

# MAGIC %md
# MAGIC ##Support Vector Machine

# COMMAND ----------

model = SVC()
model.fit(x_train, y_train)
  
predicted = model.predict(x_test)
print("The accuracy of SVM is : ", accuracy_score(y_test, predicted)*100, "%")

# COMMAND ----------

# MAGIC %md
# MAGIC ##Random Forest

# COMMAND ----------

model = RandomForestRegressor(n_estimators = 100, random_state = 0)  
model.fit(x_train, y_train)  
predicted = model.predict(x_test)
print("The accuracy of Random Forest is : ", accuracy_score(y_test, predicted.round())*100, "%")

# COMMAND ----------

# MAGIC %md
# MAGIC ##K Nearest Neighbours

# COMMAND ----------

model = KNeighborsClassifier(n_neighbors = 1)  
model.fit(x_train, y_train)
predicted = model.predict(x_test)
  

print(confusion_matrix(y_test, predicted))
print("The accuracy of KNN is : ", accuracy_score(y_test, predicted.round())*100, "%")

# COMMAND ----------

# MAGIC %md
# MAGIC ##Optimizing the KNN

# COMMAND ----------

error_rate = []
  
for i in range(1, 40):
      
    model = KNeighborsClassifier(n_neighbors = i)
    model.fit(x_train, y_train)
    pred_i = model.predict(x_test)
    error_rate.append(np.mean(pred_i != y_test))
  
plt.figure(figsize =(10, 6))
plt.plot(range(1, 40), error_rate, color ='blue',
                linestyle ='dashed', marker ='o',
         markerfacecolor ='red', markersize = 10)
  
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')

# COMMAND ----------

model = KNeighborsClassifier(n_neighbors = 7)
  
model.fit(x_train, y_train)
predicted = model.predict(x_test)
  
print('Confusion Matrix :')
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, predicted))

print()
print()
print("The accuracy of KNN is : ", accuracy_score(y_test, predicted.round())*100, "%")

# COMMAND ----------

# MAGIC %md
# MAGIC ##Gradient Boosting

# COMMAND ----------

model = xgb.XGBClassifier(use_label_encoder=False)
model.fit(x_train, y_train)
   
predicted = model.predict(x_test)
   
cm = confusion_matrix(y_test, predicted)
print()
print ("The accuracy of X Gradient Boosting is : ", accuracy_score(y_test, predicted)*100, "%")
