# Databricks notebook source
# MAGIC %md
# MAGIC # Training the Model
# MAGIC First, train a linear regression model that takes two hyperparameters: *alpha* and *l1_ratio*.
# MAGIC 
# MAGIC > The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# MAGIC > P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# MAGIC > Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

# COMMAND ----------

import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

import logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# COMMAND ----------

def eval_metrics(actual, pred):
  rmse = np.sqrt(mean_squared_error(actual, pred))
  mae = mean_absolute_error(actual, pred)
  r2 = r2_score(actual, pred)
  return rmse, mae, r2

# COMMAND ----------

try:
  alpha = float(dbutils.widgets.getArgument("alpha"))
except:
  alpha = 0.5
try:
  l1_ratio = float(dbutils.widgets.getArgument("l1_ratio"))
except:
  l1_ratio = 0.5

# COMMAND ----------

warnings.filterwarnings("ignore")
np.random.seed(40)

# Read the wine-quality csv file from the URL
csv_url =\
  'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
try:
  data = pd.read_csv(csv_url, sep=';')
except Exception as e:
  logger.exception("Unable to download training & test CSV, check your internet connection. Error: %s", e)

# Split the data into training and test sets. (0.75, 0.25) split.
train, test = train_test_split(data)

# The predicted column is "quality" which is a scalar from [3, 9]
train_x = train.drop(["quality"], axis=1)
test_x = test.drop(["quality"], axis=1)
train_y = train[["quality"]]
test_y = test[["quality"]]

# COMMAND ----------

import mlflow
import azureml.mlflow
import mlflow.sklearn
from azureml.core import Workspace
from azureml.core.authentication import ServicePrincipalAuthentication

workspace_name = dbutils.secrets.get(scope = "azureml", key = "workspace_name")
workspace_location = "eastus"
resource_group = dbutils.secrets.get(scope = "azureml", key = "resource_group")
subscription_id = dbutils.secrets.get(scope = "azureml", key = "subscription_id")

svc_pr = ServicePrincipalAuthentication(
    tenant_id = dbutils.secrets.get(scope = "azureml", key = "tenant_id"),
    service_principal_id = dbutils.secrets.get(scope = "azureml", key = "client_id"),
    service_principal_password = dbutils.secrets.get(scope = "azureml", key = "client_secret"))

workspace = Workspace.create(name = workspace_name,
                             location = workspace_location,
                             resource_group = resource_group,
                             subscription_id = subscription_id,
                             auth=svc_pr,
                             exist_ok=True)

azureml_mlflow_uri = f"azureml://{workspace_location}.api.azureml.ms/mlflow/v1.0/subscriptions/{subscription_id}/resourceGroups/{resource_group}/providers/Microsoft.MachineLearningServices/workspaces/{workspace_name}"
mlflow.set_tracking_uri(workspace.get_mlflow_tracking_uri())
mlflow.set_experiment('WineQuality')
with mlflow.start_run() as mlflow_run:
  mlflow.sklearn.autolog()
  lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
  lr.fit(train_x, train_y)

  predicted_qualities = lr.predict(test_x)

  (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

  print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
  print("  RMSE: %s" % rmse)
  print("  MAE: %s" % mae)
  print("  R2: %s" % r2)

  mlflow.log_param("alpha", alpha)
  mlflow.log_param("l1_ratio", l1_ratio)
  mlflow.log_metric("rmse", rmse)
  mlflow.log_metric("r2", r2)
  mlflow.log_metric("mae", mae)


# COMMAND ----------


import datetime
import ast,sys
input_str = sys.stdin.read()
input_list = ast.literal_eval(input_str)
dateStart=datetime.date(input_list[0],input_list[1],input_list[2])
dateEnd=datetime.date(input_list[3],input_list[4],input_list[5])
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
ans = set()
while dateStart <= dateEnd:
	ans.add(dateStart.month)
	dateStart += datetime.timedelta(1)
print([months[x-1] for x in sorted(ans)])