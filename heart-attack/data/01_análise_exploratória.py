# Databricks notebook source
from pyspark.sql import functions as F
import pandas as pd
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



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# COMMAND ----------

#%sql
#SELECT * FROM `hive_metastore`.`default`.`heart`;
df_ha = spark.read.table("hive_metastore.default.heart")
df_ha_pd = df_ha.toPandas()

df_o2sat = spark.read.table("hive_metastore.default.o_2_saturation")
df_o2sat_pd = df_o2sat.toPandas()

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT sex, count(*) FROM `hive_metastore`.`default`.`heart` GROUP BY sex;

# COMMAND ----------

display(df_ha)

# COMMAND ----------

print('Number of rows are',df_ha_pd.shape[0], 'and number of columns are ',df_ha_pd.shape[1])

# COMMAND ----------

print('Dicionárioi de dados:')
print('age - Age of the patient')
print('sex - Sex of the patient')
print('cp - Chest pain type ~ 0 = Typical Angina, 1 = Atypical Angina, 2 = Non-anginal Pain, 3 = Asymptomatic')
print('trtbps - Resting blood pressure (in mm Hg)')
print('chol - Cholestoral in mg/dl fetched via BMI sensor')
print('fbs - (fasting blood sugar > 120 mg/dl) ~ 1 = True, 0 = False')
print('restecg - Resting electrocardiographic results ~ 0 = Normal, 1 = ST-T wave normality, 2 = Left ventricular hypertrophy')
print('thalachh - Maximum heart rate achieved')
print('oldpeak - Previous peak')
print('slp - Slope')
print('caa - Number of major vessels')
print('thall - Thalium Stress Test result ~ (0,3)')
print('exng - Exercise induced angina ~ 1 = Yes, 0 = No')
print('output - Target variable')

# COMMAND ----------

df_ha_pd[df_ha_pd.duplicated()]

# COMMAND ----------

df_ha_pd.drop_duplicates(keep='first',inplace=True)

# COMMAND ----------

print('Number of rows are',df_ha_pd.shape[0], 'and number of columns are ',df_ha_pd.shape[1])

# COMMAND ----------

df_ha_pd[con_cols].describe().transpose()

# COMMAND ----------

df_ha_pd.isnull().sum()

# COMMAND ----------

dict = {}
for i in list(df_ha_pd.columns):
    dict[i] = df_ha_pd[i].value_counts().shape[0]

pd.DataFrame(dict,index=["unique count"]).transpose()

# COMMAND ----------

cat_cols = ['sex','exng','caa','cp','fbs','restecg','slp','thall']
con_cols = ["age","trtbps","chol","thalachh","oldpeak"]
target_col = ["output"]
print("Colunas de Categoria : ", cat_cols)
print("Colunas Continuas  : ", con_cols)
print("The target variable is :  ", target_col)

# COMMAND ----------

df_ha_pd.corr()

# COMMAND ----------

# MAGIC %md
# MAGIC Maiores chances de ataque cardíaco quando as variáveis abaixo estão altas:
# MAGIC cp (chest pain)/thalachh (frequencia cardiaca maxima)/slp (declive)
# MAGIC
# MAGIC Maiores chances de ataque cardíaco quando as variáveis abaixo estão baixas:
# MAGIC exng (dor no peito induzida por exercicio)/oldpeak (pico anterior)/caa (numero maior de vasos)/thall (altura)
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC Visualização dos Dados
# MAGIC

# COMMAND ----------

x=(df_ha_pd.sex.value_counts())
print(f'Número de pessoas do sexo 0 =  {x[0]} / Número de pessoas do Sexo 1 = {x[1]}')
p = sns.countplot(data=df_ha_pd, x="sex")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Podemos observar que pessoas do genero 1 é mais que o dobro do genero 0.

# COMMAND ----------

x=(df_ha_pd.cp.value_counts())
print(x)
p = sns.countplot(data=df_ha_pd, x="cp")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Podemos observar que pessoas com dor no peito do tipo 0 'Typical Angina' é maior.
# MAGIC
# MAGIC Podemos observar que pessoas com dor no peito do tipo 3 'Asymptomatic' é menor.
# MAGIC
# MAGIC Podemos observar que pessoas com dor no peito do tipo 0 é quase 50% das pessoas.
# MAGIC

# COMMAND ----------

x=(df_ha_pd.fbs.value_counts())
print(x)
p = sns.countplot(data=df_ha_pd, x="fbs")
plt.show()

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC FBS = 1 (açucar no sangue em jejum > 120 mg/dl) é bem menor que com indice menor que pessoas com FBS = 0 (abaixo de 120).
# MAGIC

# COMMAND ----------

x=(df_ha_pd.restecg.value_counts())
print(x)
p = sns.countplot(data=df_ha_pd, x="restecg")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ECG (resultados eletrocardiográficos em repouso) 
# MAGIC Valor 0: normal
# MAGIC
# MAGIC Valor 1: tem ST-T onda anormal (T inversão na onda e/ou ST elevação ou depressão de > 0.05 mV)
# MAGIC
# MAGIC Value 2: mostrando hipertrofia ventricular esquerda provável ou definitiva pelos critérios de Este
# MAGIC
# MAGIC
# MAGIC Contagem é quase a mesma para o tipo 0 e 1. Também, para o tipo 2 é quase desprezível em comparação aos tipos 0 e 1.
# MAGIC

# COMMAND ----------

x=(df_ha_pd.exng.value_counts())
print(x)
p = sns.countplot(data=df_ha_pd, x="exng")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Contagem de EXNG (dor no peito por exercício induzido = 1) é mais que o dobro para o tipo 0 (sem dor no peito).

# COMMAND ----------

x=(df_ha_pd.thall.value_counts())
print(x)
p = sns.countplot(data=df_ha_pd, x="thall")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Contagem Thall é maior para o tipo 2 e menor para o tipo 0.

# COMMAND ----------

plt.figure(figsize=(10,10))
sns.displot(df_ha_pd.age, color="red", label="Age", kde= True)
plt.legend()

# COMMAND ----------

# MAGIC %md
# MAGIC Distribuição com densidade maior no grupo etário entre 55 a 60 anos
# MAGIC

# COMMAND ----------

plt.figure(figsize=(20,20))
sns.displot(df_ha_pd.trtbps , color="green", label="Pressão Arterial em Repouso", kde= True)
plt.legend()

# COMMAND ----------

# MAGIC %md
# MAGIC Pressão Arterial em repouso tem maior contagem perto de 130

# COMMAND ----------

plt.figure(figsize=(10,10))
sns.histplot (df_ha_pd[df_ha_pd['output'] == 0]["age"], color='green',kde=True,) 
sns.histplot (df_ha_pd[df_ha_pd['output'] == 1]["age"], color='red',kde=True)
plt.title('Heart-Attack versus Idade')
plt.show()

# COMMAND ----------

plt.figure(figsize=(10,10))
sns.histplot(df_ha_pd[df_ha_pd['output'] == 0]["chol"], color='green',kde=True,) 
sns.histplot(df_ha_pd[df_ha_pd['output'] == 1]["chol"], color='red',kde=True)
plt.title('Cholestrol versus Ataques Cardíacos')
plt.show()

# COMMAND ----------

plt.figure(figsize=(10,10))
sns.histplot(df_ha_pd[df_ha_pd['output'] == 0]["trtbps"], color='green',kde=True,) 
sns.histplot(df_ha_pd[df_ha_pd['output'] == 1]["trtbps"], color='red',kde=True)
plt.title('Pressão em Repouso versus Ataques Cardíacos')
plt.show()

# COMMAND ----------

plt.figure(figsize=(10,10))
sns.histplot(df_ha_pd[df_ha_pd['output'] == 0]["thalachh"], color='green',kde=True,) 
sns.histplot(df_ha_pd[df_ha_pd['output'] == 1]["thalachh"], color='red',kde=True)
plt.title('Maior Taxa de Batimento Cardíaco versus Ataques Cardíacos')
plt.show()

# COMMAND ----------

plt.figure(figsize=(20,20))
sns.pairplot(df_ha_pd)
plt.show()

# COMMAND ----------

plt.figure(figsize=(13,13))
plt.subplot(2,3,1)
sns.violinplot(x = 'sex', y = 'output', data = df_ha_pd)
plt.subplot(2,3,2)
sns.violinplot(x = 'thall', y = 'output', data = df_ha_pd)
plt.subplot(2,3,3)
sns.violinplot(x = 'exng', y = 'output', data = df_ha_pd)
plt.subplot(2,3,4)
sns.violinplot(x = 'restecg', y = 'output', data = df_ha_pd)
plt.subplot(2,3,5)
sns.violinplot(x = 'cp', y = 'output', data = df_ha_pd)
plt.xticks(fontsize=9, rotation=45)
plt.subplot(2,3,6)
sns.violinplot(x = 'fbs', y = 'output', data = df_ha_pd)

plt.show()

# COMMAND ----------

x = df_ha_pd.iloc[:, 1:-1].values
y = df_ha_pd.iloc[:, -1].values
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

display(df_o2sat)

# COMMAND ----------

df_o2sat_pd.describe()
