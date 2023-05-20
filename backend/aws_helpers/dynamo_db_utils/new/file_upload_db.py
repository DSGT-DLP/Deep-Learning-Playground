from backend.aws_helpers.dynamo_db_utils.new.dynamo_db_utils import (
    create_dynamo_item,
    get_dynamo_item_by_id,
    update_dynamo_item,
)
import random
from datetime import datetime
from dataclasses import dataclass

TABLE_NAME = "dlp-file-upload-table"


@dataclass
class FileUploadData:
    """Data class to hold the attribute values of a record of the execution-table DynamoDB table"""

    s3_uri: str


def getFileUploadData(s3_uri: str) -> dict:
    """
    Retrieves an entry from the `execution-table` DynamoDB table given an `execution_id`. Example output: {"execution_id": "blah", "user_id": "blah", "name": "blah", "timestamp": "blah", "data_source": "TABULAR", "status": "QUEUED", "progress": 1}

    @param execution_id: The execution_id of the entry to be retrieved
    @return: A JSON string of the entry retrieved from the table
    """
    record = get_dynamo_item_by_id(TABLE_NAME, s3_uri)
    return record


def updateFileUploadData(s3_uri: str, requestData: dict) -> bool:
    """
    Updates an entry from the `execution-table` DynamoDB table given an `execution_id`.
    @param requestData: A dictionary containing the execution_id and other table attributes to be updated, with user_id as a required field
    @return a success status message if the update is successful
    """
    return update_dynamo_item(TABLE_NAME, s3_uri, requestData)


def createFileUploadData(file_upload_data: FileUploadData) -> bool:
    """
    Create a new entry corresponding to the given user_id.

    @param **kwargs: execution_id and other table attributes to be created to the new entry e.g. user_id, if does not exist
    @return: A JSON string of the entry retrieved or created from the table
    """

    return create_dynamo_item(TABLE_NAME, file_upload_data.__dict__)
