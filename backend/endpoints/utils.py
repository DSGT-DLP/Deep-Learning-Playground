import traceback

from backend.aws_helpers.dynamo_db_utils.execution_db import (
    ExecutionData,
    createUserExecutionsData,
)
from backend.common.utils import *


def createExecution(entryData: dict) -> dict:
    """
    Creates an entry in the `execution-table` DynamoDB table given an `execution_id`. If does not exist, create a new entry corresponding to the given user_id.

    E.g.
    POST request to http://localhost:8000/api/createExecution with body
    {"execution_id": "fsdh", "user_id": "fweadshas"}
    will create a new entry with the given execution_id and other attributes present e.g. user_id (user_id must be present upon creating a new entry)

    @return: A JSON string of the entry created in the table
    """
    entryData = ExecutionData(
        execution_id=entryData["execution_id"],
        user_id=entryData["user"]["uid"],
        name=entryData["custom_model_name"],
        data_source=entryData["data_source"],
        status="QUEUED",
        timestamp=get_current_timestamp(),
        progress=0,
    )
    try:
        createUserExecutionsData(entryData)
        return {"success": True, "message": "Successfully created execution entry"}
    except Exception as e:
        print(traceback.format_exc())
        return {"success": False, "message": "Error in creating execution entry"}


def send_success(results: dict):
    """
    Utility function to send success response from API with result data

    Params:
     - results (dict): Any data corresponding to API output

    Returns:
     - dict
    """
    return (json.dumps({"success": True, **results}), 200)


def send_error(message: str):
    """
    Utility function to send failure/error response from API with a custom error message

    Params:
     - message (str): error message

    Returns:
     - dict
    """
    return (json.dumps({"success": False, "message": message}), 400)


def send_train_results(train_loss_results: dict):
    """
    Wrapper function to send results of a user's training request (successful one)

    Params:
      - train_loss_results: dict containing info about training results + stats
    """
    return send_success(
        {
            "message": "Dataset trained and results outputted successfully",
            "dl_results": csv_to_json(),
            "auxiliary_outputs": train_loss_results,
        }
    )


def send_detection_results(object_detection_results: dict):
    """
    Wrapper function to send success message for object detection requests


    Args:
     - object_detection_results (dict): object detection results

    Returns:
     - dict
    """
    return send_success(
        {
            "message": "Detection worked successfully",
            "dl_results": object_detection_results["dl_results"],
            "auxiliary_outputs": object_detection_results["auxiliary_outputs"],
        }
    )


def send_traceback_error():
    """
    Wrapper function to send error messages related to code that the user may have entered during training request
    creation
    """
    return send_error(traceback.format_exc(limit=1))
