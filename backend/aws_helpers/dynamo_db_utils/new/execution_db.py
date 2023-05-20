from backend.aws_helpers.dynamo_db_utils.new.dynamo_db_utils import (
    create_dynamo_item,
    get_dynamo_item_by_id,
    get_dynamo_items_by_gsi,
    update_dynamo_item,
)
import random
from datetime import datetime
from dataclasses import dataclass

TABLE_NAME = "execution-table"


@dataclass
class ExecutionData:
    """Data class to hold the attribute values of a record of the execution-table DynamoDB table"""

    execution_id: str
    data_source: str
    metadata: dict
    name: str
    progress: int
    results: dict
    status: str
    timestamp: str
    user_id: str


def getExecutionData(execution_id: str) -> dict:
    """
    Retrieves an entry from the `execution-table` DynamoDB table given an `execution_id`. Example output: {"execution_id": "blah", "user_id": "blah", "name": "blah", "timestamp": "blah", "data_source": "TABULAR", "status": "QUEUED", "progress": 1}

    @param execution_id: The execution_id of the entry to be retrieved
    @return: A JSON string of the entry retrieved from the table
    """
    record = get_dynamo_item_by_id(TABLE_NAME, execution_id)
    return record


def updateExecutionData(execution_id: str, requestData: dict) -> bool:
    """
    Updates an entry from the `execution-table` DynamoDB table given an `execution_id`.
    @param requestData: A dictionary containing the execution_id and other table attributes to be updated, with user_id as a required field
    @return a success status message if the update is successful
    """
    return update_dynamo_item(TABLE_NAME, execution_id, requestData)


def getAllUserExecutionData(user_id: str) -> dict:
    """
    Retrieves an entry from the `execution-table` DynamoDB table given an `execution_id`. Example output: {"execution_id": "blah", "user_id": "blah", "name": "blah", "timestamp": "blah", "data_source": "TABULAR", "status": "QUEUED", "progress": 1}

    @param execution_id: The execution_id of the entry to be retrieved
    @return: A JSON string of the entry retrieved from the table
    """
    response = get_dynamo_items_by_gsi(TABLE_NAME, user_id)
    return response


def updateStatus(execution_id: str, status: str, entryData: dict = None) -> str:
    """
    Updates the status of a trainspace entry in the `trainspace` DynamoDB table given a `execution_id`. Also updates the entry with the given entryData if provided.
    @param execution_id: The execution_id of the entry to be updated
    @param status: The status to be updated
    @param entryData: The entry to be updated (if any)
    @return a success status message if the update is successful
    """
    if entryData is None:
        entryData = {}
    entryData["status"] = status
    return updateExecutionData(execution_id, entryData)


def createExecutionData(execution_data: ExecutionData) -> bool:
    """
    Create a new entry corresponding to the given user_id.

    @param **kwargs: execution_id and other table attributes to be created to the new entry e.g. user_id, if does not exist
    @return: A JSON string of the entry retrieved or created from the table
    """

    return create_dynamo_item(TABLE_NAME, execution_data.__dict__)


if __name__ == "__main__":
    print(1)
    print(2, getAllUserExecutionData("e4d46926-1eaa-42b0-accb-41a3912038e4"))
    print(3, getAllUserExecutionData("efds"))
    print(4, updateStatus("blah", "QUEUED", {"created": datetime.now().isoformat()}))
    print(
        5,
        createExecutionData(
            ExecutionData(str(random.random()), "bleh", "bleh", "bleh")
        ),
    )
