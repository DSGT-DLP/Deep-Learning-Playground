def getTrainspaceData(trainspace_id: str) -> str:
    """
    Retrieves an entry from the `execution-table` DynamoDB table given an `execution_id`. Example output: {"execution_id": "blah", "user_id": "blah", "name": "blah", "timestamp": "blah", "data_source": "TABULAR", "status": "QUEUED", "progress": 1}

    @param execution_id: The execution_id of the entry to be retrieved
    @return: A JSON string of the entry retrieved from the table
    """
    dynamoTable = TrainspaceDDBUtil(TRAINSPACE_TABLE_NAME, AWS_REGION)
    record = dynamoTable.get_record(trainspace_id)
    return json.dumps(record.__dict__)
