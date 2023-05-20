from backend.aws_helpers.dynamo_db_utils.new.dynamo_db_utils import (
    create_dynamo_item,
    get_dynamo_item_by_id,
    get_dynamo_items_by_gsi,
    update_dynamo_item,
)
import random
from datetime import datetime
from dataclasses import dataclass

TABLE_NAME = "trainspace"

@dataclass
class TrainspaceData():
    """Data class to hold the attribute values of a record of the execution-table DynamoDB table"""

    trainspace_id: str
    created: str
    data_source: str
    dataset_data: dict
    name: str
    parameters_data: dict
    review_data: str
    status: str
    uid: str


def getTrainspaceData(trainspace_id: str) -> dict[str, str or int]:
    """
    Retrieves an entry from the `execution-table` DynamoDB table given an `execution_id`. Example output: {"execution_id": "blah", "user_id": "blah", "name": "blah", "timestamp": "blah", "data_source": "TABULAR", "status": "QUEUED", "progress": 1}

    @param execution_id: The execution_id of the entry to be retrieved
    @return: A JSON string of the entry retrieved from the table
    """
    record = get_dynamo_item_by_id(TABLE_NAME, {"trainspace_id": trainspace_id})
    return record


def updateTrainspaceData(trainspace_id: str, requestData: TrainspaceData) -> bool:
    """
    Updates an entry from the `execution-table` DynamoDB table given an `execution_id`.
    @param requestData: A dictionary containing the execution_id and other table attributes to be updated, with user_id as a required field
    @return a success status message if the update is successful
    """
    return update_dynamo_item(TABLE_NAME, trainspace_id, requestData)


def getAllUserTrainspaceData(user_id: str) -> dict[str, str or int]:
    """
    Retrieves an entry from the `execution-table` DynamoDB table given an `execution_id`. Example output: {"execution_id": "blah", "user_id": "blah", "name": "blah", "timestamp": "blah", "data_source": "TABULAR", "status": "QUEUED", "progress": 1}

    @param execution_id: The execution_id of the entry to be retrieved
    @return: A JSON string of the entry retrieved from the table
    """
    response = get_dynamo_items_by_gsi(TABLE_NAME, user_id)
    return response


def updateStatus(trainspace_id: str, status: str, entryData: dict = None) -> str:
    """
    Updates the status of a trainspace entry in the `trainspace` DynamoDB table given a `trainspace_id`. Also updates the entry with the given entryData if provided.
    @param trainspace_id: The trainspace_id of the entry to be updated
    @param status: The status to be updated
    @param entryData: The entry to be updated (if any)
    @return a success status message if the update is successful
    """
    if entryData is None:
        entryData = {}
    entryData["status"] = status
    return updateTrainspaceData(trainspace_id, entryData)

def createTrainspaceData(trainspace_data: TrainspaceData) -> bool:
    """
    Create a new entry corresponding to the given user_id.

    @param **kwargs: execution_id and other table attributes to be created to the new entry e.g. user_id, if does not exist
    @return: A JSON string of the entry retrieved or created from the table
    """

    return create_dynamo_item(TABLE_NAME, trainspace_data.__dict__)


if __name__ == "__main__":
    print(1)
    print(2, getAllUserTrainspaceData("e4d46926-1eaa-42b0-accb-41a3912038e4"))
    print(3, getAllUserTrainspaceData("efds"))
    print(4, updateStatus("blah", "QUEUED", {"created": datetime.now().isoformat()}))
    print(5, createTrainspaceData(TrainspaceData(str(random.random()), "bleh", "bleh", "bleh")))
