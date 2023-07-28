from aws_helpers.dynamo_db_utils.constants import (
    TRAINSPACE_TABLE_NAME,
    TrainStatus,
)
from aws_helpers.dynamo_db_utils.dynamo_db_utils import (
    create_dynamo_item,
    get_dynamo_item_by_key,
    get_dynamo_items_by_gsi,
    update_dynamo_item,
)
import random
from datetime import datetime
from dataclasses import dataclass

TABLE_NAME = TRAINSPACE_TABLE_NAME


@dataclass
class TrainspaceData:
    """Data class to hold the attribute values of a record of the trainspace DynamoDB table"""

    trainspace_id: str
    uid: str
    created: str = ""
    data_source: str = ""
    dataset_data: dict = None
    name: str = ""
    parameters_data: dict = None
    review_data: str = ""
    status: str = TrainStatus.QUEUED.name


def getTrainspaceData(trainspace_id: str) -> dict:
    """
    Retrieves an entry from the `trainspace` DynamoDB table given an `trainspace_id`. Example output: {"trainspace_id": "blah", "user_id": "blah", "name": "blah", "timestamp": "blah", "data_source": "TABULAR", "status": "QUEUED", "progress": 1}

    @param trainspace_id: The trainspace_id of the entry to be retrieved
    @return: A JSON string of the entry retrieved from the table
    """
    record = get_dynamo_item_by_key(TABLE_NAME, trainspace_id)
    return record


def updateTrainspaceData(trainspace_id: str, requestData: dict) -> bool:
    """
    Updates an entry from the `trainspace` DynamoDB table given an `trainspace_id`.

    @param trainspace_id: The trainspace_id of the entry to be updated
    @param requestData: A dictionary containing the other table attributes to be updated, not including trainspace_id
    @return True if the update is successful
    """
    return update_dynamo_item(TABLE_NAME, trainspace_id, requestData)


def getAllUserTrainspaceData(user_id: str) -> list[dict]:
    """
    Retrieves all entries of this user from the `trainspace` DynamoDB table given an `trainspace_id`. Example output: [{"trainspace_id": "blah", "user_id": "blah", "name": "blah", "timestamp": "blah", "data_source": "TABULAR", "status": "QUEUED", "progress": 1}]

    @param trainspace_id: The trainspace_id of the entry to be retrieved
    @return: A list of all matching entries retrieved from the table
    """
    response = get_dynamo_items_by_gsi(TABLE_NAME, user_id)
    return response


def updateStatus(trainspace_id: str, status: str, entryData: dict = None) -> bool:
    """
    Updates the status of a trainspace entry in the `trainspace` DynamoDB table given a `trainspace_id`. Also updates the entry with the given entryData if provided.

    @param trainspace_id: The trainspace_id of the entry to be updated
    @param status: The status to be updated
    @param entryData: The entry to be updated (if any)
    @return True if the update is successful
    """
    if entryData is None:
        entryData = {}
    # entryData["status"] = status
    return updateTrainspaceData(trainspace_id, entryData)


def createTrainspaceData(trainspace_data: TrainspaceData) -> bool:
    """
    Create a new entry or replaces an existing entry table according to the `trainspace_id`.

    @param trainspace_data: trainspace_id and other table attributes to be created or updated if the entry already exists
    @return: True if the creation or update is successful
    """

    return create_dynamo_item(TABLE_NAME, trainspace_data.__dict__)
