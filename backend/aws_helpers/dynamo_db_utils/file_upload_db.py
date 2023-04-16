import json

from dataclasses import dataclass
from datetime import datetime

from backend.aws_helpers.dynamo_db_utils.base_db import (
    BaseData,
    BaseDDBUtil,
    enumclass,
    changevar,
)
from backend.common.constants import FILE_UPLOAD_TABLE_NAME, AWS_REGION
from boto3.dynamodb.conditions import Key
from backend.common.utils import get_current_timestamp, get_current_unix_time
from typing import Union

PRIMARY_KEY = "s3_uri"

REQUIRED_KEYS = [
    "s3_uri",
    "uid",
    "created",
    "filename",
]


@dataclass
class FileUploadData(BaseData):
    """Data class to hold the attribute values of a record of the dlp-file-upload-table DynamoDB table"""

    s3_uri: str
    uid: str
    created: str
    filename: str
    ttl: int = None


@enumclass(
    DataClass=FileUploadData,
)
class FileUploadEnums:
    """Class that holds the enums associated with the FileUploadDDBUtil class. It includes:
    ExecutionEnums.Attribute - Enum that defines the schema of the dlp-file-upload-table. It holds the attribute names of the table
    """

    pass


@changevar(
    DataClass=FileUploadData, EnumClass=FileUploadEnums, partition_key=PRIMARY_KEY
)
class FileUploadDDBUtil(BaseDDBUtil):
    """Class that interacts with AWS DynamoDB to manipulate information stored in the dlp-file-upload-table DynamoDB table"""

    pass


def getFileUploadData(s3_uri: str) -> str:
    """
    Retrieves an entry from the `dlp-file-upload-table` DynamoDB table given an `s3_uri`. 
    Example output: {"s3_uri": "s3://upload-bucket/blah/blah.csv", "uid": "blah", "filename": "blah.csv", "created": "blah"}
    @param s3_uri: The s3_uri of the file that we care about
    @return: A JSON string of the entry retrieved from the table
    """
    dynamoTable = FileUploadDDBUtil(FILE_UPLOAD_TABLE_NAME, AWS_REGION)
    record = dynamoTable.get_record(s3_uri)
    return json.dumps(record.__dict__)


def createFileUploadData(entryData: dict) -> str:
    """
    Create a new entry corresponding to the given s3_uri.
    @param **kwargs: s3_uri and other table attributes to be created to the new entry
    @return: A JSON string of the entry retrieved or created from the table
    """
    if not validate_keys(entryData, REQUIRED_KEYS):
        raise ValueError(f"Missing keys {REQUIRED_KEYS} in request body")

    dynamoTable = FileUploadDDBUtil(FILE_UPLOAD_TABLE_NAME, AWS_REGION)
    entryData["ttl"] = get_current_unix_time() + 5*24*60*60 # add time to live
    newRecord = FileUploadData(**entryData)
    dynamoTable.create_record(newRecord)
    return json.dumps(newRecord.__dict__)


def updateFileUploadData(requestData: dict) -> str:
    """
    Updates an entry from the `dlp-file-upload-table` DynamoDB table given an `s3_uri`.
    @param requestData: A dictionary containing the s3_uri and other table attributes to be updated
    @return a success status message if the update is successful
    """
    if not validate_keys(requestData, REQUIRED_KEYS):
        raise ValueError(f"Missing keys {REQUIRED_KEYS} in request body")

    dynamoTable = FileUploadDDBUtil(FILE_UPLOAD_TABLE_NAME, AWS_REGION)
    s3_uri = requestData[PRIMARY_KEY]
    updatedRecord = FileUploadData(**requestData).__dict__
    updatedRecord.pop(PRIMARY_KEY)
    dynamoTable.update_record(s3_uri, **updatedRecord)
    return '{"status": "success"}'


def getAllUserFileUploadData(uid: str) -> str:
    """
    Retrieves an entry from the `dlp-file-upload-table` DynamoDB table given an `uid`. Example output: {"execution_id": "blah", "user_id": "blah", "name": "blah", "timestamp": "blah", "data_source": "TABULAR", "status": "QUEUED", "progress": 1}
    @param uid: The user_id of the entry to be retrieved
    @return: A JSON string of the entry retrieved from the table
    """
    dynamoTable = FileUploadDDBUtil(FILE_UPLOAD_TABLE_NAME, AWS_REGION)
    response = dynamoTable.table.query(
        IndexName="uid", KeyConditionExpression=Key("uid").eq(uid)
    )
    items = response["Items"]
    record = []
    for item in items:
        record.append(
            {
                "s3_uri": item["s3_uri"],
                "uid": item["uid"],
                "created": item["created"],
                "filename": item["filename"],
            }
        )

    while "LastEvaluatedKey" in response:
        key = response["LastEvaluatedKey"]
        response = dynamoTable.table.query(
            KeyConditionExpression=Key("uid").eq(uid), ExclusiveStartKey=key
        )
        for item in items:
            record.append(
                {
                    "s3_uri": item["s3_uri"],
                    "uid": item["uid"],
                    "created": item["created"],
                    "filename": item["filename"],
                }
            )
    return json.dumps(record)


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