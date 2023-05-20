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
class TrainspaceData:
    """Data class to hold the attribute values of a record of the trainspace DynamoDB table"""

    trainspace_id: str
    created: str
    data_source: str
    dataset_data: dict
    name: str
    parameters_data: dict
    review_data: str
    status: str
    uid: str


def getTrainspaceData(trainspace_id: str) -> dict:
    """
    Retrieves an entry from the `trainspace` DynamoDB table given an `trainspace_id`. Example output: {"trainspace_id": "blah", "user_id": "blah", "name": "blah", "timestamp": "blah", "data_source": "TABULAR", "status": "QUEUED", "progress": 1}

    @param trainspace_id: The trainspace_id of the entry to be retrieved
    @return: A JSON string of the entry retrieved from the table
    """
    record = get_dynamo_item_by_id(TABLE_NAME, trainspace_id)
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


def updateStatus(trainspace_id: str, status: str, entryData: dict = None) -> str:
    """
    Updates the status of a trainspace entry in the `trainspace` DynamoDB table given a `trainspace_id`. Also updates the entry with the given entryData if provided.

    @param trainspace_id: The trainspace_id of the entry to be updated
    @param status: The status to be updated
    @param entryData: The entry to be updated (if any)
    @return True if the update is successful
    """
    if entryData is None:
        entryData = {}
    entryData["status"] = status
    return updateTrainspaceData(trainspace_id, entryData)


def createTrainspaceData(trainspace_data: TrainspaceData) -> bool:
    """
    Create a new entry or replaces an existing entry table according to the `trainspace_id`.

    @param trainspace_data: trainspace_id and other table attributes to be created or updated if the entry already exists
    @return: True if the creation or update is successful
    """

    return create_dynamo_item(TABLE_NAME, trainspace_data.__dict__)


if __name__ == "__main__":
    print(1)
    print(2, getTrainspaceData("e4d46926-1eaa-42b0-accb-41a3912038e4"))
    print(3, getAllUserTrainspaceData("efds"))
    print(4, updateStatus("blah", "QUEUED", {"created": datetime.now().isoformat()}))
    print(
        5,
        createTrainspaceData(
            TrainspaceData(
                trainspace_id=str(random.random()),
                created=datetime.now().isoformat(),
                data_source="TABULAR",
                dataset_data={
                    "name": {"S": "IRIS"},
                    "is_default_dataset": {"BOOL": True},
                },
                name=str(random.random()),
                parameters_data={
                    "features": {
                        "L": [
                            {"S": "sepal length (cm)"},
                            {"S": "sepal width (cm)"},
                            {"S": "petal length (cm)"},
                            {"S": "petal width (cm)"},
                        ]
                    },
                    "criterion": {"S": "CELOSS"},
                    "batch_size": {"N": "20"},
                    "test_size": {"N": "0.2"},
                    "target_col": {"S": "target"},
                    "layers": {
                        "L": [
                            {
                                "M": {
                                    "value": {"S": "LINEAR"},
                                    "parameters": {"L": [{"N": "10"}, {"N": "3"}]},
                                }
                            },
                            {"M": {"value": {"S": "RELU"}, "parameters": {"L": []}}},
                            {
                                "M": {
                                    "value": {"S": "LINEAR"},
                                    "parameters": {"L": [{"N": "3"}, {"N": "10"}]},
                                }
                            },
                            {
                                "M": {
                                    "value": {"S": "SOFTMAX"},
                                    "parameters": {"L": [{"N": "-1"}]},
                                }
                            },
                        ]
                    },
                    "problem_type": {"S": "CLASSIFICATION"},
                    "shuffle": {"BOOL": True},
                    "epochs": {"N": "5"},
                    "optimizer_name": {"S": "SGD"},
                },
                review_data={
                    "notification_phone_number": {"NULL": True},
                    "notification_email": {"NULL": True},
                },
                status="QUEUED",
                uid="bleh",
            )
        ),
    )
