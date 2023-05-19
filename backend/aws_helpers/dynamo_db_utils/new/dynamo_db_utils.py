import boto3
from backend.aws_helpers.dynamo_db_utils.new.constants import ALL_DYANMODB_TABLES
from datetime import datetime

dynamodb = boto3.resource('dynamodb')

def get_dynamo_item(table_name: str, keys: dict[str, str or int]) -> dict:
    """
    Get item from DynamoDB table that match all keys, raises exception if item does not exist

    Args:
        table_name (str): Name of DynamoDB table
        key (str): Keys of item to get, e.g., {'id': 'fedfsd', 'name': 'bob'}
    
    Returns:
        dict: Item from DynamoDB table in JSON format

    Raises:
        ValueError: If table_name is not a valid table name
        Exception: If DynamoDB get_item call fails for any reason, e.g., item not found
    """
    # Validation steps
    if table_name not in ALL_DYANMODB_TABLES.keys():
        raise ValueError("Invalid table name: " + table_name)
    
    # Get item
    table = dynamodb.Table(table_name)
    response = table.get_item(Key=keys)
    if response.get('Item') is None:
        raise Exception("Item not found")
    
    return response['Item']


def create_dynamo_item(table_name: str, input_item: dict) -> bool:
    """
    Creates item in DynamoDB table, replaces item if partition_key already exists

    Args:
        table_name (str): Name of DynamoDB table
        input_item (dict): object to insert into table in the form of a dictionary e.g. {'id': 'fesd', 'name': 'bob'}. Values can only be strings or numbers
    
    Returns:
        dict: Item from DynamoDB table in JSON format

    Raises:
        ValueError: If table_name is not a valid table name
        Exception: If DynamoDB get_item call fails for any reason, e.g., item not found
    """
    # Validation steps
    if table_name not in ALL_DYANMODB_TABLES.keys():
        raise ValueError("Invalid table name: " + table_name)
    if input_item.get(ALL_DYANMODB_TABLES[table_name]['partition_key']) is None:
        raise ValueError("Item must have the partition key")

    item = { k: v for k, v in input_item.items() }
    # Create item
    table = dynamodb.Table(table_name)
    response = table.put_item(Item=item)
    if (response['ResponseMetadata']['HTTPStatusCode'] != 200):
        raise Exception("Failed to delete item")
    return True


def update_dynamo_item(table_name: str, input_item: dict) -> bool:
    """
    Updates item in DynamoDB table, creates if partition_key does not exist

    Args:
        table_name (str): Name of DynamoDB table
        input_item (dict): object to insert into table in the form of a dictionary e.g. {'id': 'fesd', 'name': 'bob'}. Values can only be strings or numbers
    
    Returns:
        dict: Item from DynamoDB table in JSON format

    Raises:
        ValueError: If table_name is not a valid table name
        Exception: If DynamoDB get_item call fails for any reason, e.g., item not found
    """
    return create_dynamo_item(table_name, input_item)


def delete_dynamo_item(table_name: str, keys: dict[str, str or int]) -> bool:
    """
    Deletes item from DynamoDB table by matching given keys, does nothing if item does not exist

    Args:
        table_name (str): Name of DynamoDB table
        key (str): Key of item to delete

    Returns:
        true if the item was created successfully

    Raises:
        ValueError: If table_name is not a valid table name
        Exception: If DynamoDB delete_item call fails for any reason
    """
    # Validation steps
    if table_name not in ALL_DYANMODB_TABLES.keys():
        raise ValueError("Invalid table name: " + table_name)
    
    # Delete item
    table = dynamodb.Table(table_name)
    response = table.delete_item(Key=keys)
    if (response['ResponseMetadata']['HTTPStatusCode'] != 200):
        raise Exception("Failed to delete item")
    return True



if __name__ == "__main__":
    print(1)
    # print(delete_dynamo_item("trainspace", {"trainspace_id": "blah"}))
    # exit()
    print(update_dynamo_item("trainspace", 
                                      {"trainspace_id": "blah", "uid": "bleh", "created": datetime.now().isoformat()}))
