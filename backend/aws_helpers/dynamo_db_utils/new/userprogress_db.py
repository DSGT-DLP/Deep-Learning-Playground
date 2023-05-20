from backend.aws_helpers.dynamo_db_utils.new.dynamo_db_utils import (
    create_dynamo_item,
    get_dynamo_item_by_id,
    update_dynamo_item,
)
import random
from datetime import datetime
from dataclasses import dataclass

TABLE_NAME = "userprogress_table"


@dataclass
class UserProgressData:
    """Data class to hold the attribute values of a record of the execution-table DynamoDB table"""

    uid: str
    progressData: dict


def getUserProgressData(uid: str) -> dict:
    """
    Retrieves an entry from the `execution-table` DynamoDB table given an `uid`. Example output: {"uid": "blah", "uid": "blah", "name": "blah", "timestamp": "blah", "data_source": "TABULAR", "status": "QUEUED", "progress": 1}

    @param uid: The uid of the entry to be retrieved
    @return: A JSON string of the entry retrieved from the table
    """
    record = get_dynamo_item_by_id(TABLE_NAME, uid)
    return record


def updateUserProgressData(uid: str, requestData: dict) -> bool:
    """
    Updates an entry from the `execution-table` DynamoDB table given an `uid`.
    @param requestData: A dictionary containing the uid and other table attributes to be updated, with uid as a required field
    @return a success status message if the update is successful
    """
    return update_dynamo_item(TABLE_NAME, uid, requestData)


def createUserProgressData(execution_data: UserProgressData) -> bool:
    """
    Create a new entry corresponding to the given uid.

    @param **kwargs: uid and other table attributes to be created to the new entry e.g. uid, if does not exist
    @return: A JSON string of the entry retrieved or created from the table
    """

    return create_dynamo_item(TABLE_NAME, execution_data.__dict__)


if __name__ == "__main__":
    print(1)
    print(2, getAllUserUserProgressData("e4d46926-1eaa-42b0-accb-41a3912038e4"))
    print(3, getAllUserUserProgressData("efds"))
    print(4, updateStatus("blah", "QUEUED", {"created": datetime.now().isoformat()}))
    print(
        5,
        createUserProgressData(
            UserProgressData(str(random.random()), "bleh", "bleh", "bleh")
        ),
    )
