import mlflow
import azureml.mlflow
import mlflow.tensorflow
from azureml.core import Workspace
from azureml.core.authentication import ServicePrincipalAuthentication


workspace_name = "amlworkspace"

#workspace_name = dbutils.secrets.get(scope = "azureml", key = "workspace_name")
workspace_location = "eastus"
resource_group = "DatabricksMLflow"
subscription_id = "a9381c9a-fc83-487d-92ca-7aff1ea7d216"
#resource_group = dbutils.secrets.get(scope = "azureml", key = "resource_group")
#subscription_id = dbutils.secrets.get(scope = "azureml", key = "subscription_id")

svc_pr = ServicePrincipalAuthentication(
    #tenant_id = dbutils.secrets.get(scope = "azureml", key = "tenant_id")
    tenant_id = "ccae492a-41d1-486b-9665-6483c17bb8a6",
    #service_principal_id =  dbutils.secrets.get(scope = "azureml", key = "client_id"),
    #App ID
    service_principal_id = "472e99db-d062-4f9a-9efa-c721ed09861e",
    #service_principal_password = dbutils.secrets.get(scope = "azureml", key = "client_secret")
    service_principal_password = "zR48Q~nvxvFu_vciHnNyfr7bq5Zph05c2o2XKb9Z"
    )

workspace = Workspace.create(name = workspace_name,
                             location = workspace_location,
                             resource_group = resource_group,
                             subscription_id = subscription_id,
                             auth=svc_pr,
                             exist_ok=True)

#azureml_mlflow_uri = #f"azureml://{workspace_location}.api.azureml.ms/mlflow/v1.0/subscriptions/{subscription_id}/resourceGroups/{resource_group}/providers/Microsoft.MachineLearningServices/works#paces/{workspace_name}"
mlflow.set_tracking_uri(workspace.get_mlflow_tracking_uri())
mlflow.set_experiment('TestFirs2')