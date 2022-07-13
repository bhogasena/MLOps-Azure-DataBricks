import mlflow
import json
import mlflow.tensorflow
from azureml.core import Workspace
from azureml.core.authentication import ServicePrincipalAuthentication
from mlflow.tracking.client import MlflowClient
from mlflow.entities import ViewType
from mlflow.deployments import get_deploy_client
from  azureml.core import Webservice
from azureml.core.webservice import AciWebservice
from azureml.core import Model


workspace_name = "amlworkspace"
workspace_location = "eastus"
resource_group = "DatabricksMLflow"
subscription_id = "a9381c9a-fc83-487d-92ca-7aff1ea7d216"

svc_pr = ServicePrincipalAuthentication(
    tenant_id = "ccae492a-41d1-486b-9665-6483c17bb8a6",
    service_principal_id = "472e99db-d062-4f9a-9efa-c721ed09861e",
    service_principal_password = "zR48Q~nvxvFu_vciHnNyfr7bq5Zph05c2o2XKb9Z"
    )

ws = Workspace.create(name = workspace_name,
                             location = workspace_location,
                             resource_group = resource_group,
                             subscription_id = subscription_id,
                             auth=svc_pr,
                             exist_ok=True)

mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
experiment = MlflowClient().get_experiment_by_name('TestFirsRavi')                             

client = get_deploy_client(mlflow.get_tracking_uri())

# set the model path 
model_path = "tensorflow-model"
runs = MlflowClient().search_runs(experiment.experiment_id)
runid=None
for run in runs:
    runid=run.info.run_id
print(runid)

deploy_config ={
      "computeType": "aci",
      "containerResourceRequirements":
         {
            "cpu": 1,
            "memoryInGB": 1
        },
        "location": "eastus",
        "authEnabled": True
}

# Write the deployment configuration into a file.
deployment_config_path = "deployment_config.json"
with open(deployment_config_path, "w") as outfile:
    outfile.write(json.dumps(deploy_config))


config = {"deploy-config-file": deployment_config_path}

try:
    client.delete_deployment("mlflow-keras-deploy")
    client.create_deployment(name="mlflow-keras-deploy", model_uri='runs:/{}/{}'.format(runid, model_path), config=config,endpoint='mlflow-keras-ep')
except:
    client.create_deployment(name="mlflow-keras-deploy", model_uri='runs:/{}/{}'.format(runid, model_path), config=config,endpoint='mlflow-keras-ep')


service = AciWebservice(workspace=ws, name='mlflow-keras-deploy')
scoringuri = service.scoring_uri
print(service.get_keys())


