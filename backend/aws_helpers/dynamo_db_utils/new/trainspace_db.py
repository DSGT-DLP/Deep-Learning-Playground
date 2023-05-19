from backend.aws_helpers.dynamo_db_utils.new.dynamo_db_utils import get_dynamo_item, create_dynamo_item, update_dynamo_item, delete_dynamo_item

TABLE_NAME = 'trainspace'

def getTrainspaceData(trainspace_id: str) -> dict:
    """
    Retrieves an entry from the `execution-table` DynamoDB table given an `execution_id`. Example output: {"execution_id": "blah", "user_id": "blah", "name": "blah", "timestamp": "blah", "data_source": "TABULAR", "status": "QUEUED", "progress": 1}

    @param execution_id: The execution_id of the entry to be retrieved
    @return: A JSON string of the entry retrieved from the table
    """
    record = get_dynamo_item(TABLE_NAME, {'trainspace_id': trainspace_id})
    return record


def updateTrainspaceData(requestData: dict) -> bool:
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

