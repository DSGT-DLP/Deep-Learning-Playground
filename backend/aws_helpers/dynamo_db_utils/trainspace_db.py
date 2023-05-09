import json

from dataclasses import dataclass
from datetime import datetime

from backend.aws_helpers.dynamo_db_utils.base_db import (
    BaseData,
    BaseDDBUtil,
    enumclass,
    changevar,
)
from backend.common.constants import TRAINSPACE_TABLE_NAME, AWS_REGION
from boto3.dynamodb.conditions import Key
from backend.common.utils import get_current_timestamp
from typing import Union

PRIMARY_KEY = "trainspace_id"

REQUIRED_KEYS = [
    "trainspace_id",
    "uid",
]


@dataclass
class TrainspaceData(BaseData):
    """Data class to hold the attribute values of a record of the execution-table DynamoDB table"""

    trainspace_id: str
    uid: str
    dataset_data: str
    status: str
    created: str
    modified: str
    train_model: str
    train_parameters: str = None
    train_results: str = None


@enumclass(
    DataClass=TrainspaceData,
    train_model=[
        "TABULAR",
        "PRETRAINED",
        "IMAGE",
        "AUDIO",
        "TEXTUAL",
        "CLASSICAL_ML",
        "OBJECT_DETECTION",
    ],
    step=["UPLOAD_FILE", "PREPROCESS", "TRAIN", "RESULTS"],
    status=["QUEUED", "STARTING", "UPLOADING", "TRAINING", "SUCCESS", "ERROR"],
)
class TrainspaceEnums:
    """Class that holds the enums associated with the ExecutionDDBUtil class. It includes:
    ExecutionEnums.Attribute - Enum that defines the schema of the execution-table. It holds the attribute names of the table
    ExecutionEnums.Execution_Source - Enum that defines the categorical values associated with the 'execution_source' attribute
    """

    pass


@changevar(
    DataClass=TrainspaceData, EnumClass=TrainspaceEnums, partition_key=PRIMARY_KEY
)
class TrainspaceDDBUtil(BaseDDBUtil):
    """Class that interacts with AWS DynamoDB to manipulate information stored in the execution-table DynamoDB table"""

    pass


def getTrainspaceData(trainspace_id: str) -> str:
    """
    Retrieves an entry from the `execution-table` DynamoDB table given an `execution_id`. Example output: {"execution_id": "blah", "user_id": "blah", "name": "blah", "timestamp": "blah", "data_source": "TABULAR", "status": "QUEUED", "progress": 1}

    @param execution_id: The execution_id of the entry to be retrieved
    @return: A JSON string of the entry retrieved from the table
    """
    dynamoTable = TrainspaceDDBUtil(TRAINSPACE_TABLE_NAME, AWS_REGION)
    record = dynamoTable.get_record(trainspace_id)
    return json.dumps(record.__dict__)


def createTrainspaceData(entryData: dict) -> str:
    """
    Create a new entry corresponding to the given user_id.

    @param **kwargs: execution_id and other table attributes to be created to the new entry e.g. user_id, if does not exist
    @return: A JSON string of the entry retrieved or created from the table
    """
    if not validate_keys(entryData, REQUIRED_KEYS):
        raise ValueError(f"Missing keys {REQUIRED_KEYS} in request body")

    dynamoTable = TrainspaceDDBUtil(TRAINSPACE_TABLE_NAME, AWS_REGION)
    newRecord = TrainspaceData(**entryData)
    dynamoTable.create_record(newRecord)
    return json.dumps(newRecord.__dict__)


def updateTrainspaceData(requestData: dict) -> str:
    """
    Updates an entry from the `execution-table` DynamoDB table given an `execution_id`.
    @param requestData: A dictionary containing the execution_id and other table attributes to be updated, with user_id as a required field
    @return a success status message if the update is successful
    """
    if not validate_keys(requestData, REQUIRED_KEYS):
        raise ValueError(f"Missing keys {REQUIRED_KEYS} in request body")

    dynamoTable = TrainspaceDDBUtil(TRAINSPACE_TABLE_NAME, AWS_REGION)
    trainspace_id = requestData[PRIMARY_KEY]
    updatedRecord = TrainspaceData(**requestData).__dict__
    updatedRecord.pop(PRIMARY_KEY)
    dynamoTable.update_record(trainspace_id, **updatedRecord)
    return '{"status": "success"}'


def getAllUserTrainspaceData(user_id: str) -> str:
    """
    Retrieves an entry from the `execution-table` DynamoDB table given an `execution_id`. Example output: {"execution_id": "blah", "user_id": "blah", "name": "blah", "timestamp": "blah", "data_source": "TABULAR", "status": "QUEUED", "progress": 1}

    @param execution_id: The execution_id of the entry to be retrieved
    @return: A JSON string of the entry retrieved from the table
    """
    dynamoTable = TrainspaceDDBUtil(TRAINSPACE_TABLE_NAME, AWS_REGION)
    response = dynamoTable.table.query(
        IndexName="uid", KeyConditionExpression=Key("uid").eq(user_id)
    )
    items = response["Items"]
    record = []
    for item in items:
        record.append(
            {
                "trainspace_id": item["trainspace_id"],
                "name": item["name"],
                "training_file": item["training_file"],
                "uid": item["uid"],
                "step": item["step"],
                "status": item["status"],
                "created": item["created"],
                "modified": item["modified"],
                "train_model": item["train_model"],
                "train_parameters": item["train_parameters"],
                "train_results": item["train_results"],
            }
        )

    while "LastEvaluatedKey" in response:
        key = response["LastEvaluatedKey"]
        response = dynamoTable.table.query(
            KeyConditionExpression=Key("uid").eq(user_id), ExclusiveStartKey=key
        )
        for item in items:
            record.append(
                {
                    "trainspace_id": item["trainspace_id"],
                    "name": item["name"],
                    "training_file": item["training_file"],
                    "uid": item["uid"],
                    "step": item["step"],
                    "status": item["status"],
                    "created": item["created"],
                    "modified": item["modified"],
                    "train_model": item["train_model"],
                    "train_parameters": item["train_parameters"],
                    "train_results": item["train_results"],
                }
            )
    return json.dumps(record)


def updateStatus(trainspace_id: str, status: str, entryData: dict = None) -> str:
    """
    Updates the status of an entry from the `execution-table` DynamoDB table given an `execution_id`.

    @param execution_id: The execution_id of the entry to be updated
    @param status: The new status of the entry
    @return a success status message if the update is successful
    """
    try:
        dynamoTable = TrainspaceDDBUtil(TRAINSPACE_TABLE_NAME, AWS_REGION)
        dynamoTable.update_record(
            trainspace_id, status=status, modified=get_current_timestamp()
        )
        return '{"status": "success"}'
    except:
        entryData["status"] = status
        createTrainspaceData(entryData)
        return '{"status": "success"}'


def validate_keys(requestData: dict, required_keys: list[str]) -> bool:
    """
    Validates all required_keys in the requestData dictionary are present.

    @param requestData: The dictionary to be checked
    @param required_keys: The list of keys to be checked
    @return a boolean value indicating whether all required_keys are present in the requestData dictionary
    """
    for key in required_keys:
        if key not in requestData.keys():
            return False
    return True
