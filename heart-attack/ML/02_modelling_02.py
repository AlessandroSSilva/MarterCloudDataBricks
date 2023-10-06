# Databricks notebook source
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

# COMMAND ----------

#%sql
#SELECT * FROM `hive_metastore`.`default`.`heart`;
df_Spark = spark.read.table("hive_metastore.default.heart_attack_gold")
df = df_Spark.toPandas()



# COMMAND ----------

# MAGIC %md
# MAGIC ##Modeling

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Variáveis relevantes: sex, age, cp, thall, oldpeak, exng, restecg, caa

# COMMAND ----------

import mlflow
import mlflow.lightgbm

# COMMAND ----------

mlflow.end_run()

# COMMAND ----------

mlflow.start_run()

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
#df1 = df1.drop(['output'],axis=1)
#df1 = df1.drop(['trtbps'],axis=1)
#df1 = df1.drop(['chol'],axis=1)
#df1 = df1.drop(['fbs'],axis=1)

# define the columns to be encoded and scaled
cat_cols = ['sex','exng','caa','cp','fbs','restecg','slp','thall']
con_cols = ["age","trtbps","chol","thalachh","oldpeak"]

#cat_cols = ['sex','exng','caa','cp','restecg','slp','thall']
#con_cols = ["age","thalachh","oldpeak"]

target_col = ["output"]
print("The categorial cols are : ", cat_cols)
print("The continuous cols are : ", con_cols)
print("The target variable is :  ", target_col)

# encoding the categorical columns
df1 = pd.get_dummies(df1, columns = cat_cols, drop_first = True)

# defining the features and target

X = df1.drop(['output'],axis=1)
#X = df1.drop(['trtbps'],axis=1)
#X = df1.drop(['chol'],axis=1)
#X = df1.drop(['fbs'],axis=1)

y = df1[['output']]

# instantiating the scaler
scaler = RobustScaler()

# scaling the continuous featuree
X[con_cols] = scaler.fit_transform(X[con_cols])
print("The first 5 rows of X are")
X.head()

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# COMMAND ----------

cat_cols = ['sex','exng','caa','cp','fbs','restecg','slp','thall']
con_cols = ["age","trtbps","chol","thalachh","oldpeak"]

#cat_cols = ['sex','exng','caa','cp','restecg','slp','thall']
#con_cols = ["age","thalachh","oldpeak"]

target_col = ["output"]
print("The categorial cols are : ", cat_cols)
print("The continuous cols are : ", con_cols)
print("The target variable is :  ", target_col)

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)
print("The shape of X_train is      ", X_train.shape)
print("The shape of X_test is       ",X_test.shape)
print("The shape of y_train is      ",y_train.shape)
print("The shape of y_test is       ",y_test.shape)

# COMMAND ----------

# instantiate the classifier
gbt = GradientBoostingClassifier(n_estimators = 300,max_depth=1,subsample=0.8,max_features=0.2,random_state=42)

# fitting the model
gbt.fit(X_train,y_train)

# predicting values
y_pred = gbt.predict(X_test)
print("The test accuracy score of Gradient Boosting Classifier is ", accuracy_score(y_test, y_pred))

# COMMAND ----------

# MAGIC %md
# MAGIC ###Linear Classifiers

# COMMAND ----------

# instantiating the object and fitting
clf = SVC(kernel='linear', C=1, random_state=42).fit(X_train,y_train)
clf.fit(X_train, y_train)
# predicting the values
y_pred = clf.predict(X_test)

# printing the test accuracy
print("The test accuracy score of SVM is ", accuracy_score(y_test, y_pred))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Registrar Modelo Linear

# COMMAND ----------

# instantiate the classifier
gbt = GradientBoostingClassifier(n_estimators = 300,max_depth=1,subsample=0.8,max_features=0.2,random_state=42)

# fitting the model
gbt.fit(X_train,y_train)

# predicting values
y_pred = gbt.predict(X_test)
print("The test accuracy score of Gradient Boosting Classifier is ", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# COMMAND ----------

# clf = lgb.LGBMClassifier(objective='multiclass', class_weight='balanced')
# clf = lgb.LGBMRegressor(objective='multiclass', class_weight='balanced')

# COMMAND ----------

# Define numerical features (all features in this case)
##numerical_features = X.columns.tolist()

# Create a numerical transformer with StandardScaler
##numerical_transformer = Pipeline(steps=[
##    ('scaler', StandardScaler())
##])

# Create a preprocessor
##preprocessor = ColumnTransformer(
##    transformers=[
##        ('num', numerical_transformer, numerical_features)
##    ])

# COMMAND ----------

##pipeline = Pipeline(steps=[
##    ('preprocessor', preprocessor),
##    ('classifier', clf)
##])#

# COMMAND ----------

##pipeline

# COMMAND ----------

##pipeline.fit(X_train, y_train)

# COMMAND ----------

##y_pred = pipeline.predict(X_test)

# COMMAND ----------

# antigo
##print(classification_report(y_test, y_pred))

# COMMAND ----------

# novo
##print(classification_report(y_test, y_pred))

# COMMAND ----------

##conf_matrix = confusion_matrix(y_test, y_pred)

# COMMAND ----------

##sns.heatmap(conf_matrix, annot=True, fmt="d")

# COMMAND ----------

# MAGIC %md
# MAGIC ###Hyperparameter tuning of SVC

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

print(classification_report(y_test, y_pred))

# COMMAND ----------

# MAGIC %md
# MAGIC ###Logistic Regression

# COMMAND ----------

# instantiating the object
logreg = LogisticRegression()

# fitting the object
logreg.fit(X_train, y_train)

# calculating the probabilities
y_pred_proba = logreg.predict_proba(X_test)

# finding the predicted valued
y_pred = np.argmax(y_pred_proba,axis=1)

# printing the test accuracy
print("The test accuracy score of Logistric Regression is ", accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))

# COMMAND ----------

# MAGIC %md
# MAGIC ###ROC Curve

# COMMAND ----------

# calculating the probabilities
y_pred_prob = logreg.predict_proba(X_test)[:,1]

# instantiating the roc_cruve
fpr,tpr,threshols=roc_curve(y_test,y_pred_prob)

# plotting the curve
plt.plot([0,1],[0,1],"k--",'r+')
plt.plot(fpr,tpr,label='Logistic Regression')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Logistric Regression ROC Curve")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ###Tree Models

# COMMAND ----------

# instantiating the object
dt = DecisionTreeClassifier(random_state = 42)

# fitting the model
dt.fit(X_train, y_train)

# calculating the predictions
y_pred = dt.predict(X_test)

# printing the test accuracy
print("The test accuracy score of Decision Tree is ", accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))

# COMMAND ----------

# MAGIC %md
# MAGIC ###Random Forest

# COMMAND ----------

# instantiating the object
rf = RandomForestClassifier()

# fitting the model
rf.fit(X_train, y_train)

# calculating the predictions
y_pred = dt.predict(X_test)

# printing the test accuracy
print("The test accuracy score of Random Forest is ", accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))

# COMMAND ----------

# MAGIC %md
# MAGIC ###Gradient Boosting Classifier - without tuning

# COMMAND ----------

# instantiate the classifier
gbt = GradientBoostingClassifier(n_estimators = 300,max_depth=1,subsample=0.8,max_features=0.2,random_state=42)

# fitting the model
gbt.fit(X_train,y_train)

# predicting values
y_pred = gbt.predict(X_test)
print("The test accuracy score of Gradient Boosting Classifier is ", accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))

# COMMAND ----------

# MAGIC %md
# MAGIC #Registrar Modelo

# COMMAND ----------

mlflow.lightgbm.log_model(
lgb_model=clf,
artifact_path="lightgbm-model",
registered_model_name="classificador-potencial-heart-attack",
)

# COMMAND ----------

mlflow.end_run()

# COMMAND ----------

X_train.iloc[0].to_dict()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Variáveis importantes

# COMMAND ----------

#feature_importances = pipeline.named_steps['classifier'].feature_importances_
