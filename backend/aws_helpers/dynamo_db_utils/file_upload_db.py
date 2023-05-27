from backend.aws_helpers.dynamo_db_utils.constants import FILE_UPLOAD_TABLE_NAME
from backend.aws_helpers.dynamo_db_utils.dynamo_db_utils import (
    create_dynamo_item,
    get_dynamo_item_by_id,
    get_dynamo_items_by_gsi,
    update_dynamo_item,
)
import random
from datetime import datetime
from dataclasses import dataclass

TABLE_NAME = FILE_UPLOAD_TABLE_NAME


@dataclass
class FileUploadData:
    """Data class to hold the attribute values of a record of the dlp-file-upload-table DynamoDB table"""

    s3_uri: str
    uid: str


def getFileUploadData(s3_uri: str) -> dict:
    """
    Retrieves an entry from the `dlp-file-upload-table` DynamoDB table given an `s3_uri`

    @param s3_uri: The s3_uri of the entry to be retrieved
    @return: A JSON string of the entry retrieved from the table
    """
    record = get_dynamo_item_by_id(TABLE_NAME, s3_uri)
    return record


def updateFileUploadData(s3_uri: str, requestData: dict) -> bool:
    """
    Updates an entry from the `dlp-file-upload-table` DynamoDB table given an `s3_uri`.

    @param s3_uri: The s3_uri of the entry to be updated
    @param requestData: A dictionary containing the other table attributes to be updated, not including s3_uri
    @return a success status message if the update is successful
    """
    return update_dynamo_item(TABLE_NAME, s3_uri, requestData)


def createFileUploadData(file_upload_data: FileUploadData) -> bool:
    """
    Create a new entry or replaces an existing entry table according to the `s3_uri`.

    @param file_upload_data: s3_uri and other table attributes to be created or updated if the entry already exists
    @return: True if the creation or update is successful
    """

    return create_dynamo_item(TABLE_NAME, file_upload_data.__dict__)


def getAllUserFileUploadData(uid: str) -> list[dict]:
    """
    Retrieves an entry from the `dlp-file-upload-table` DynamoDB table given an `uid`. Example output: {"execution_id": "blah", "user_id": "blah", "name": "blah", "timestamp": "blah", "data_source": "TABULAR", "status": "QUEUED", "progress": 1}
    @param uid: The user_id of the entry to be retrieved
    @return: A JSON string of the entry retrieved from the table
    """
    response = get_dynamo_items_by_gsi(TABLE_NAME, uid)
    return response
