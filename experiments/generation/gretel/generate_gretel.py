import os
import yaml
import gzip
import shutil
import tarfile
import urllib.request
from pathlib import Path

from syntherela.metadata import Metadata
from syntherela.data import load_tables, remove_sdv_columns, save_tables

from gretel_client import configure_session
from gretel_client import create_or_get_unique_project
from gretel_client.config import get_session_config
from gretel_client.rest_v1.api.connections_api import ConnectionsApi
from gretel_client.rest_v1.api.workflows_api import WorkflowsApi
from gretel_client.rest_v1.models import (
    CreateWorkflowRunRequest,
    CreateWorkflowRequest,
)
from gretel_client.workflows.logs import print_logs_for_workflow_run

import argparse

args = argparse.ArgumentParser()
args.add_argument("--dataset-name", type=str, default="Biodegradability_v1")
args.add_argument("--real-data-path", type=str, default="data/original")
args.add_argument("--synthetic-data-path", type=str, default="data/synthetic")
args.add_argument("--connection-uid", type=str, required=True)
args.add_argument("--model", type=str, required=True, choices=["lstm", "actgan"])
args.add_argument("--run-id", type=str, default="1")
args = args.parse_args()

dataset_name = args.dataset_name
real_data_path = args.real_data_path
synthetic_data_path = args.synthetic_data_path
input_connection_uid = args.connection_uid
model = args.model
run_id = args.run_id

# Helpers for running workflows from the notebook


def run_workflow(config: str):
    """Create a workflow, and workflow run from a given yaml config. Blocks and
    prints log lines until the workflow reaches a terminal state.

    Args:
        config: The workflow config to run.
    """
    print("Validating actions in the config...")
    config_dict = yaml.safe_load(config)

    for action in config_dict["actions"]:
        print(f"Validating action {action['name']}")
        response = workflow_api.validate_workflow_action(action)
        print(f"Validation response: {response}")

    workflow = workflow_api.create_workflow(
        CreateWorkflowRequest(
            project_id=project.project_guid,
            config=config_dict,
            name=config_dict["name"],
        )
    )

    workflow_run = workflow_api.create_workflow_run(
        CreateWorkflowRunRequest(workflow_id=workflow.id)
    )

    print(f"workflow: {workflow.id}")
    print(f"workflow run id: {workflow_run.id}")

    print_logs_for_workflow_run(workflow_run.id, session)


# Log into Gretel
configure_session(api_key="prompt", cache="yes", validate=True)


# Load real data
metadata = Metadata().load_from_json(
    Path(real_data_path) / f"{dataset_name}/metadata.json"
)
real_data = load_tables(Path(real_data_path) / f"{dataset_name}", metadata)
real_data, metadata = remove_sdv_columns(real_data, metadata)
metadata.validate_data(real_data)

# ## Designate Project for your Relational Workflow
table_names = list(real_data.keys())
dataset_name_gretel = dataset_name.replace("_", "-")

session = get_session_config()
connection_api = session.get_v1_api(ConnectionsApi)
workflow_api = session.get_v1_api(WorkflowsApi)

project = create_or_get_unique_project(name=f"Synthesize-{dataset_name_gretel}-{model}")

# Configure and Run your Relational Workflow
# Gretel Workflows provide an easy to use, config driven API for automating and operationalizing synthetic data. A Gretel Workflow is constructed by actions that are composed to create a pipeline for processing data with Gretel. To learn more about Gretel Workflows, check out [our docs](https://docs.gretel.ai/reference/workflows).

### Define Source Data via Connector
connection_type = connection_api.get_connection(input_connection_uid).dict()["type"]

# ### Define Workflow configuration
workflow_config = f"""\
name: {dataset_name_gretel}-{connection_type}-workflow-{model}

actions:
  - name: {connection_type}-read
    type: {connection_type}_source
    connection: {input_connection_uid}
    config:
      sync:
        mode: full

  - name: model-train-run
    type: gretel_tabular
    input: {connection_type}-read
    config:
      project_id: {project.project_guid}
      train:
        model: "synthetics/tabular-{model}"
        dataset: "{{{connection_type}-read.outputs.dataset}}"
      run:
        num_records_multiplier: 1.0

"""
print(workflow_config)


# ### Run Workflow

run_workflow(workflow_config)

# ## View Results

path = f"./data_{dataset_name}_{model}"
os.makedirs(path, exist_ok=True)

# Download the output artifacts
urllib.request.urlretrieve(
    project.get_artifact_link(project.artifacts[-1]["key"]),
    f"./data_{dataset_name}_{model}/workflow-output.tar.gz",
)

with gzip.open(f"./data_{dataset_name}_{model}/workflow-output.tar.gz", "rb") as f_in:
    with open(f"./data_{dataset_name}_{model}/workflow-output.tar", "wb") as f_out:
        f_out.write(f_in.read())

with tarfile.open(f"{path}/workflow-output.tar") as tar:
    tar.extractall(path)


path_synthetic = (
    f"{synthetic_data_path}/{dataset_name}/GRETEL_{model.upper()}/{run_id}/sample1"
)
os.makedirs(path_synthetic, exist_ok=True)
for table in table_names:
    shutil.copy(f"{path}/synth_{table}.csv", f"{path_synthetic}/{table}.csv")


# Postprocess synthetic data (some categorical columns are generated as floats)
def is_float(value):
    if value is None or value == "":
        return False
    try:
        float(value)
        return True
    except:  # noqa: E722
        return False


synthetic_data = load_tables(Path(path_synthetic), metadata)
metadata.validate_data(synthetic_data)

for table in metadata.get_tables():
    table_meta = metadata.get_table_meta(table)
    for column, column_info in table_meta["columns"].items():
        if column_info["sdtype"] == "categorical":
            values = synthetic_data[table][column].unique()
            numeric = False
            for value in values:
                if is_float(str(value)) and value == value:
                    numeric = True
                    break
            if numeric:
                synthetic_data[table][column] = synthetic_data[table][column].astype(
                    "object"
                )
                for i, value in synthetic_data[table][column].items():
                    if value != value or not is_float(str(value)):
                        # skip NaN
                        continue
                    synthetic_data[table].at[i, column] = int(float(value))
                synthetic_data[table][column] = synthetic_data[table][column].astype(
                    "object"
                )

# Save synthetic data
save_tables(synthetic_data, Path(path_synthetic))
