import json

from dataclasses import dataclass
from datetime import datetime

from backend.aws_helpers.dynamo_db_utils.base_db import BaseData, BaseDDBUtil, enumclass, changevar
from backend.common.constants import EXECUTION_TABLE_NAME, AWS_REGION
from typing import Union

@dataclass
class ExecutionData(BaseData):
    """Data class to hold the attribute values of a record of the execution-table DynamoDB table"""
    execution_id: str
    user_id: str = None
    name: str = None
    timestamp: str = None
    data_source: str = None
    status: str = None
    progress: int = None
    
@enumclass(
    DataClass=ExecutionData,
    data_source=['TABULAR', 'PRETRAINED', 'IMAGE', 'AUDIO', 'TEXTUAL'],
    status=['QUEUED', 'STARTING', 'UPLOADING', 'TRAINING', 'SUCCESS', 'ERROR']
)
class ExecutionEnums:
    """Class that holds the enums associated with the ExecutionDDBUtil class. It includes:
        ExecutionEnums.Attribute - Enum that defines the schema of the execution-table. It holds the attribute names of the table
        ExecutionEnums.Execution_Source - Enum that defines the categorical values associated with the 'execution_source' attribute"""
    pass

@changevar(DataClass=ExecutionData, EnumClass=ExecutionEnums, partition_key='execution_id')
class ExecutionDDBUtil(BaseDDBUtil):
    """Class that interacts with AWS DynamoDB to manipulate information stored in the execution-table DynamoDB table"""
    pass

def getUserExecutionsData(execution_id: str) -> str:
    """
    Retrieves an entry from the `execution-table` DynamoDB table given an `execution_id`. Example output: {"execution_id": "blah", "user_id": "blah", "name": "blah", "timestamp": "blah", "data_source": "TABULAR", "status": "QUEUED", "progress": 1}

    @param execution_id: The execution_id of the entry to be retrieved
    @return: A JSON string of the entry retrieved from the table
    """
    dynamoTable = ExecutionDDBUtil(EXECUTION_TABLE_NAME, AWS_REGION)
    record = dynamoTable.get_record(execution_id)
    return json.dumps(record.__dict__)

def createUserExecutionsData(entryData: dict) -> str:
    """
    Create a new entry corresponding to the given user_id.

    @param **kwargs: execution_id and other table attributes to be created to the new entry e.g. user_id, if does not exist
    @return: A JSON string of the entry retrieved or created from the table
    """
    if not validate_keys(entryData, ["user_id", "execution_id"]):
        raise ValueError("Invalid keys in request body")

    dynamoTable = ExecutionDDBUtil(EXECUTION_TABLE_NAME, AWS_REGION)
    newRecord = ExecutionData(**entryData)
    dynamoTable.create_record(newRecord)
    return json.dumps(newRecord.__dict__)

def updateUserExecutionsData(requestData: dict) -> str:
    """
    Updates an entry from the `execution-table` DynamoDB table given an `execution_id`.

    @param requestData: A dictionary containing the execution_id and other table attributes to be updated, with user_id as a required field
    @return a success status message if the update is successful
    """

    execution_id = requestData["execution_id"]
    dataInput = {
        "data_source": requestData.get("data_source", None),
        "name": requestData.get("name", None),
        "progress": requestData.get("progress", None),
        "status": requestData.get("status", None),
        "timestamp": requestData.get("timestamp", None),
        "user_id": requestData["user_id"]
    }

    dynamoTable = ExecutionDDBUtil(EXECUTION_TABLE_NAME, AWS_REGION)
    dynamoTable.update_record(execution_id, **dataInput)
    return "{\"status\": \"success\"}"

def validate_keys(requestData: dict, required_keys: list[str]) -> bool:
    """
    Validates all required_keys in the requestData dictionary are present.

    @param requestData: The dictionary to be checked
    @param required_keys: The list of keys to be checked
    @return a boolean value indicating whether all required_keys are present in the requestData dictionary
    """
    for key in requestData.keys():
        if key not in valid_keys:
            return False
    return True
