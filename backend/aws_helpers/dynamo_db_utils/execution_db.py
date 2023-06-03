from backend.aws_helpers.dynamo_db_utils.constants import EXECUTION_TABLE_NAME
from backend.aws_helpers.dynamo_db_utils.dynamo_db_utils import (
    create_dynamo_item,
    get_dynamo_item_by_key,
    get_dynamo_items_by_gsi,
    update_dynamo_item,
)
import random
from datetime import datetime
from dataclasses import dataclass

TABLE_NAME = EXECUTION_TABLE_NAME


@dataclass
class ExecutionData:
    """Data class to hold the attribute values of a record of the execution-table DynamoDB table"""

    execution_id: str
    data_source: str
    name: str
    status: str
    timestamp: str
    user_id: str
    progress: int = 0
    results: dict = None
    metadata: dict = None


def getExecutionData(execution_id: str) -> dict:
    """
    Retrieves an entry from the `execution-table` DynamoDB table given an `execution_id`. Example output: {"execution_id": "blah", "user_id": "blah", "name": "blah", "timestamp": "blah", "data_source": "TABULAR", "status": "QUEUED", "progress": 1}

    @param execution_id: The execution_id of the entry to be retrieved
    @return: A JSON dict of the entry retrieved from the table
    """
    record = get_dynamo_item_by_key(TABLE_NAME, execution_id)
    return record


def updateExecutionData(execution_id: str, requestData: dict) -> bool:
    """
    Updates an entry from the `execution-table` DynamoDB table given an `execution_id`.

    @param execution_id: The execution_id of the entry to be updated
    @param requestData: A dictionary containing the table attributes to be updated, not including execution_id
    @return a success status message if the update is successful
    """
    return update_dynamo_item(TABLE_NAME, execution_id, requestData)


def getAllUserExecutionData(user_id: str) -> list[dict]:
    """
    Retrieves all entries of this user from the `execution-table` DynamoDB table given an `execution_id`. Example output: [{"execution_id": "blah", "user_id": "blah", "name": "blah", "timestamp": "blah", "data_source": "TABULAR", "status": "QUEUED", "progress": 1}]

    @param user_id: The user_id of the entries to be retrieved
    @return: A list of all matching entries retrieved from the table
    """
    response = get_dynamo_items_by_gsi(TABLE_NAME, user_id)
    return response


def updateStatus(execution_id: str, status: str, entryData: dict = None) -> bool:
    """
    Updates the status of a trainspace entry in the `trainspace` DynamoDB table given a `execution_id`. Also updates the entry with the given entryData if provided.

    @param execution_id: The execution_id of the entry to be updated
    @param status: The status to be updated
    @param entryData: The entry to be updated (if any)
    @return True if the update is successful
    """
    if entryData is None:
        entryData = {}
    entryData["status"] = status
    return updateExecutionData(execution_id, entryData)


def createExecutionData(execution_data: ExecutionData) -> bool:
    """
    Create a new entry corresponding to the given user_id. Replaces any existing entry with the same execution_id.

    @param execution_data: execution_id and other table attributes to be created to the new entry e.g. user_id, if does not exist
    @return: True if the creation is successful
    """

    return create_dynamo_item(TABLE_NAME, execution_data.__dict__)
