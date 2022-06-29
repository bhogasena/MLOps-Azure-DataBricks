# Databricks notebook source
# MAGIC %md ## Inference

# COMMAND ----------

import mlflow
import azureml.mlflow
from azureml.core import Workspace
from azureml.core.authentication import ServicePrincipalAuthentication
from mlflow.deployments import get_deploy_client
from mlflow.tracking.client import MlflowClient

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
client = get_deploy_client(mlflow.get_tracking_uri())

# COMMAND ----------

from mlflow.entities import ViewType

experiment_name = "WineQuality"
experiment = MlflowClient().get_experiment_by_name(experiment_name)

query = "metrics.rmse < 0.8"
runs = MlflowClient().search_runs(experiment.experiment_id, query, ViewType.ALL)

rmse_low = None
run_id = None
for run in runs:
  if (rmse_low == None or run.data.metrics['rmse'] < rmse_low):
    rmse_low = run.data.metrics['rmse']
    run_id = run.info.run_id
print("Lowest RMSE:", rmse_low)
print("Run ID:", run_id)

model_uri = "runs:/" + run_id + "/model"

# COMMAND ----------



# COMMAND ----------

# MAGIC %md ## Load MLflow Model as a scikit-learn Model
# MAGIC You can use the MLflow API to load the model from the MLflow server that was produced by a given run.
# MAGIC 
# MAGIC Once you load it, it is a just a scikit-learn model and you can explore it or use it.

# COMMAND ----------

import mlflow.sklearn
model = mlflow.sklearn.load_model(model_uri=model_uri)
model.coef_

# COMMAND ----------

import numpy as np
import pandas as pd

cols = ['alcohol', 'chlorides', 'citric acid', 'density', 'fixed acidity', 'free sulfur dioxide', 'pH', 'residual sugar', 'sulphates', 'total sulfur dioxide', 'volatile acidity']
d = [12.8, 0.029, 0.48, 0.98, 6.2, 29, 3.33, 1.2, 0.39, 75, 0.66]
d = np.array([d])

data = pd.DataFrame(d, columns=cols)
display(data)

# COMMAND ----------

#Get a prediction for a row of the dataset
model.predict(data)

# COMMAND ----------

# MAGIC %md ## Use an MLflow Model for Batch Inference
# MAGIC You can get a PySpark UDF to do some batch inference using one of the models.

# COMMAND ----------

csv_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
try:
  data = pd.read_csv(csv_url, sep=';')
except Exception as e:
  logger.exception("Unable to download training & test CSV, check your internet connection. Error: %s", e)
  
# Create a Spark DataFrame from the original pandas DataFrame minus the column you want to predict.
# Use this to simulate what this would be like if you had a big data set e.g. click logs that was 
# regularly being updated that you wanted to score.
dataframe = spark.createDataFrame(data.drop(["quality"], axis=1))
display(dataframe)

# COMMAND ----------

# MAGIC %md Use the MLflow API to create a PySpark UDF from a run. See [Export a python_function model as an Apache Spark UDF](https://mlflow.org/docs/latest/models.html#export-a-python-function-model-as-an-apache-spark-udf).

# COMMAND ----------

import mlflow.pyfunc
pyfunc_udf = mlflow.pyfunc.spark_udf(spark, model_uri=model_uri,env_manager="conda")

# COMMAND ----------

# MAGIC %md Add a column to the data by applying the PySpark UDF to the DataFrame.

# COMMAND ----------

predicted_df = dataframe.withColumn("prediction", pyfunc_udf('alcohol', 'chlorides', 'citric acid', 'density', 'fixed acidity', 'free sulfur dioxide', 'pH', 'residual sugar', 'sulphates', 'total sulfur dioxide', 'volatile acidity'))
display(predicted_df)
