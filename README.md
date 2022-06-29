# Introduction 
This Projects has details on implemeting MLOps with Azure Devops Pipelines and MLFlow to train your modesl on Azure Databricks and log them onto Azure ML workspace and deploy onto Azure ML Real time end point.

# Getting Started

### Prerequisites

1. Azure Subscription.
2. Azure Devops Organization Account.
3. PIP Installation.


### Installation

1. Create Azure Databricks and Azure ML WorkSaces and Link both of them in Azure Portal.
2. Generate User Access Token in Azure Databricks workspace. This token will be used to connect to Azure Databricks host from local system and Azure Devops pipeline variables.
3. Login to Azure Cloud Shell and Create Service Prinical to use for Authentication with Azure ML workspace.
	```sh
	az ad sp create-for-rbac -n <ServicePrincialName> --role contributor --scope "/subscriptions/<SubscriptionId>/resourceGroups/<ResourceGroupName>"
	```
	Note down the AppId(i.e, Client_Id) and Secret.
4. Install Databricks CLI in your local system.
	```sh
	pip install databricks-cli
	```
5. Setup Databricks Authentication.
	```sh
	databricks configure --token
	```
	
	Enter Databricks URL which you can copy from Azure Databricks workspace.
	Token is the one you generated in Step2.
	
6.  Create Secret Scope in Databricks
	```sh
	databricks secrets create-scope --scope azureml
	```
7. Add Secrets to this scope ex: client_id, client_secret of Service Prinicpal.
	```sh
	databricks secrets put --scope azureml --key client_id --string-value <secretvalue>
	```
8. Azure devops pipeline yml is provided in this repo to create Pipeline in Azure Devops.
	

# Build and Test

1. You can run the Notebooks in Azure DataBricks Workspace manually and which will log the experiments and Models in Azure ML workspace.

Below are explanations for the Code snippets present in the Notebooks.

1. Below is the Code to connect to Azure ML Workspace and Set the MLFlow Tracking URI of Azure ML Workspace.
	```sh
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
    ```
	
2. Create the Experiment in Azure ML Work space.
	```sh
	mlflow.set_experiment('WineQuality')
	```
3. You can enable the Autolog using below code. In below example used sklearn module.
	```sh
	mlflow.sklearn.autolog()
	```
4. You can log any parameters you want.
	```sh
	  mlflow.log_param("alpha", alpha)
	  mlflow.log_param("l1_ratio", l1_ratio)
	  mlflow.log_metric("rmse", rmse)
	  mlflow.log_metric("r2", r2)
	  mlflow.log_metric("mae", mae)
	  ```
5. Get the MLFlow Client .
	```sh
	client = get_deploy_client(mlflow.get_tracking_uri())
	```
6. Deloy Model.
	```sh
	client.create_deployment(
    model_uri=model_uri,
    config=config,
    name="winequality-model",
    endpoint='winequality-endpoint'
    )
	```

	
