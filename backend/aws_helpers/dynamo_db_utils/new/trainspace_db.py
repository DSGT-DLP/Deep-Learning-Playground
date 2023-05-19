from backend.aws_helpers.dynamo_db_utils.new.dynamo_db_utils import get_dynamo_item, create_dynamo_item, update_dynamo_item, delete_dynamo_item
from datetime import datetime

TABLE_NAME = 'trainspace'

def getTrainspaceData(trainspace_id: str) -> dict[str, str or int]:
    """
    Retrieves an entry from the `execution-table` DynamoDB table given an `execution_id`. Example output: {"execution_id": "blah", "user_id": "blah", "name": "blah", "timestamp": "blah", "data_source": "TABULAR", "status": "QUEUED", "progress": 1}

    @param execution_id: The execution_id of the entry to be retrieved
    @return: A JSON string of the entry retrieved from the table
    """
    record = get_dynamo_item(TABLE_NAME, {'trainspace_id': trainspace_id})
    return record


def updateTrainspaceData(trainspace_id: str, requestData: dict) -> bool:
    """
    Updates an entry from the `execution-table` DynamoDB table given an `execution_id`.
    @param requestData: A dictionary containing the execution_id and other table attributes to be updated, with user_id as a required field
    @return a success status message if the update is successful
    """
    return update_dynamo_item(TABLE_NAME, {'trainspace_id': trainspace_id},requestData)


def getAllUserTrainspaceData(user_id: str) -> dict[str, str or int]:
    """
    Retrieves an entry from the `execution-table` DynamoDB table given an `execution_id`. Example output: {"execution_id": "blah", "user_id": "blah", "name": "blah", "timestamp": "blah", "data_source": "TABULAR", "status": "QUEUED", "progress": 1}

    @param execution_id: The execution_id of the entry to be retrieved
    @return: A JSON string of the entry retrieved from the table
    """
    response = get_dynamo_item(TABLE_NAME, {'uid': user_id})
    return response

def updateStatus(trainspace_id: str, status: str, entryData: dict = None) -> str:
    """
    Updates the status of a trainspace entry in the `trainspace` DynamoDB table given a `trainspace_id`. Also updates the entry with the given entryData if provided.
    @param trainspace_id: The trainspace_id of the entry to be updated
    @param status: The status to be updated
    @param entryData: The entry to be updated
    @return a success status message if the update is successful
    """
    if entryData is None:
        entryData = {}
    entryData['status'] = status
    return updateTrainspaceData(trainspace_id, entryData)


if __name__ == "__main__":
    print(1)
    print(getAllUserTrainspaceData("bleh"))
    exit()
    print(updateStatus("blah", "TRAINING", {
                                          "created": datetime.now().isoformat()}))
