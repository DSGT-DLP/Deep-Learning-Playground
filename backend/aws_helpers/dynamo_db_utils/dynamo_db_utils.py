from copy import deepcopy
from decimal import Decimal
import boto3
from backend.aws_helpers.dynamo_db_utils.constants import ALL_DYANMODB_TABLES
from backend.common.constants import AWS_REGION
import random
from datetime import datetime
from typing import Union

dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)


def get_dynamo_item_by_key(table_name: str, partition_key_value: str) -> dict:
    """
    Get item from DynamoDB table that match the partition key, raises exception if item does not exist

    Args:
        table_name (str): Name of DynamoDB table
        partition_key_value (str): Value of partition key to match

    Returns:
        dict: One item from DynamoDB table in JSON format

    Raises:
        ValueError: If table_name is not a valid table name
        Exception: If DynamoDB get_item call fails for any reason, e.g., item not found
    """
    # Validation steps
    if table_name not in ALL_DYANMODB_TABLES.keys():
        raise ValueError("Invalid table name: " + table_name)

    # Get item
    partition_key = ALL_DYANMODB_TABLES[table_name]["partition_key"]
    item_key = {partition_key: partition_key_value}
    table = dynamodb.Table(table_name)
    response = table.get_item(Key=item_key)
    if response.get("Item") is None:
        raise Exception("Item not found")

    return response["Item"]


def get_dynamo_items_by_gsi(table_name: str, gsi_value: Union[int, str]) -> list[dict]:
    """
    Get items from DynamoDB table that match the given GSI value, returns [] if no items match

    Args:
        table_name (str): Name of DynamoDB table
        gsi_value (int or str): Value of GSI to match

    Returns:
        list[dict]: One array containing all matching items from DynamoDB table in JSON format

    Raises:
        ValueError: If table_name is not a valid table name
        Exception: If DynamoDB get_item call fails for any reason
    """
    # Validation steps
    if table_name not in ALL_DYANMODB_TABLES.keys():
        raise ValueError("Invalid table name: " + table_name)
    gsi_key = ALL_DYANMODB_TABLES[table_name].get("gsi")
    if gsi_key is None:
        raise Exception("Table does not have a GSI")

    # Make query params
    PREFIX = "#dlp__"

    # create update expression
    key_condition_expression = f"{PREFIX}{gsi_key} = :{gsi_key}"

    # create expression attribute values
    expression_attribute_values = {f":{gsi_key}": gsi_value}

    # create expression attribute names to prevent reserved words error
    expression_attribute_names = {f"{PREFIX}{gsi_key}": gsi_key}

    # initialize response items
    response_items = []
    table = dynamodb.Table(table_name)
    has_more_items = True
    last_evaluated_key = None

    # Get items through pagination
    while has_more_items:
        query_params = {
            "IndexName": gsi_key,
            "KeyConditionExpression": key_condition_expression,
            "ExpressionAttributeValues": expression_attribute_values,
            "ExpressionAttributeNames": expression_attribute_names,
        }
        if last_evaluated_key is not None:
            query_params["ExclusiveStartKey"] = last_evaluated_key

        response = table.query(**query_params)
        response_items.extend(response["Items"])
        last_evaluated_key = response.get("LastEvaluatedKey")
        has_more_items = last_evaluated_key is not None

    return response_items


def to_decimal(o):
    if isinstance(o, dict):
        for k, v in o.items():
            if isinstance(v, float):
                o[k] = Decimal(str(v))
            else:
                to_decimal(v)
    elif isinstance(o, list):
        for v in o:
            to_decimal(v)


def create_dynamo_item_from_obj(table_name: str, obj: object) -> bool:
    obj_dict = deepcopy(obj.__dict__)
    to_decimal(obj_dict)
    return create_dynamo_item(table_name, obj_dict)


def create_dynamo_item(table_name: str, input_item: dict) -> bool:
    """
    Creates item in DynamoDB table, replaces item if partition_key already exists

    Args:
        table_name (str): Name of DynamoDB table
        input_item (dict): object to insert into table in the form of a dictionary e.g. {'id': 'fesd', 'name': 'bob'}.

    Returns:
        bool: True if item was created successfully

    Raises:
        ValueError: If table_name is not a valid table name, or the input dict is missing the partition key or gsi key (if it exists)
        Exception: If DynamoDB get_item call fails for any reason, e.g., item not found
    """
    # Validation steps
    if table_name not in ALL_DYANMODB_TABLES.keys():
        raise ValueError("Invalid table name: " + table_name)
    partition_key = ALL_DYANMODB_TABLES[table_name]["partition_key"]
    if input_item.get(partition_key) is None:
        raise ValueError("Item must have the partition key: " + partition_key)
    gsi_key = ALL_DYANMODB_TABLES[table_name].get("gsi")
    if gsi_key and input_item.get(gsi_key) is None:
        raise ValueError("Item must have the gsi key: " + gsi_key)

    # Create item
    table = dynamodb.Table(table_name)
    response = table.put_item(Item=input_item)
    if response["ResponseMetadata"]["HTTPStatusCode"] != 200:
        raise Exception("Failed to create item")
    return True


def update_dynamo_item(
    table_name: str, partition_key_value: str or int, update_object: dict
) -> bool:
    """
    Updates item in DynamoDB table, creates if partition_key does not exist

    Args:
        table_name (str): Name of DynamoDB table
        partition_key_value (str): Value of partition key to match
        update_object (dict): dict containing the columns and values to update into table in the form of a dictionary e.g. {'name': 'bob'}, must not contain partition key

    Returns:
        bool: True if item was updated successfully

    Raises:
        ValueError: If table_name is not a valid table name
        Exception: If DynamoDB get_item call fails for any reason, e.g., item not found
    """
    # Validation steps
    if table_name not in ALL_DYANMODB_TABLES.keys():
        raise ValueError("Invalid table name: " + table_name)
    partition_key = ALL_DYANMODB_TABLES[table_name]["partition_key"]
    if partition_key in update_object.keys():
        raise ValueError("Cannot update partition key: " + partition_key)

    PREFIX = "#dlp__"

    # create update expression
    update_expression = "SET " + ", ".join(
        [f"{PREFIX}{key} = :{key}" for key in update_object.keys()]
    )

    # create expression attribute values
    expression_attribute_values = {
        f":{key}": value for key, value in update_object.items()
    }

    # create expression attribute names to prevent reserved words error
    expression_attribute_names = {f"{PREFIX}{key}": key for key in update_object.keys()}

    # Create item
    partition_key = ALL_DYANMODB_TABLES[table_name]["partition_key"]
    table = dynamodb.Table(table_name)
    response = table.update_item(
        Key={partition_key: partition_key_value},
        UpdateExpression=update_expression,
        ExpressionAttributeValues=expression_attribute_values,
        ExpressionAttributeNames=expression_attribute_names,
    )
    if response["ResponseMetadata"]["HTTPStatusCode"] != 200:
        raise Exception("Failed to delete item")
    return True


def delete_dynamo_item(table_name: str, partition_key_value: Union[int, str]) -> bool:
    """
    Deletes item from DynamoDB table by matching given keys, does nothing if item does not exist

    Args:
        table_name (str): Name of DynamoDB table
        partition_key_value (str): Partition key value of item to delete

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
    partition_key = ALL_DYANMODB_TABLES[table_name]["partition_key"]
    table = dynamodb.Table(table_name)
    response = table.delete_item(Key={partition_key: partition_key_value})
    if response["ResponseMetadata"]["HTTPStatusCode"] != 200:
        raise Exception("Failed to delete item")
    return True
