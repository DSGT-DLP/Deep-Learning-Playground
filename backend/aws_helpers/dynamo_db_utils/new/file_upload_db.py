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
    """Data class to hold the attribute values of a record of the dlp-file-upload-table DynamoDB table"""

    s3_uri: str


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
