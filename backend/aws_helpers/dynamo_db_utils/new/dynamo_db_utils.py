import boto3
from backend.aws_helpers.dynamo_db_utils.new.constants import ALL_DYANMODB_TABLES
import random
from datetime import datetime

dynamodb = boto3.client('dynamodb')

class DynamoDbUtils():
    def __init__(self):
        pass

    def get_item(self, table_name: str, partition_key: str) -> dict:
        """
        Get item from DynamoDB table by key

        Args:
            table_name (str): Name of DynamoDB table
            key (str): Key of item to get
        
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
        item_key = {
            ALL_DYANMODB_TABLES[table_name]['partition_key']: {'S': partition_key}
        }
        response = dynamodb.get_item(TableName=table_name, Key=item_key)
        if response.get('Item') is None:
            raise Exception("Item not found")
        
        return response['Item']
    

    def create_item(self, table_name: str, input_item: dict) -> dict:
        """
        Get item from DynamoDB table by key

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

        item = dict()
        for key in input_item.keys():
            value = input_item[key]
            if type(value) == str:
                item[key] = {'S': value}
            elif type(value) == float or type(value) == int:
                item[key] = {'N': value}
            else:
                raise ValueError("Only number and strings accepted for dynamoDB. Invalid value for key: " + key)
        # Create item
        response = dynamodb.put_item(TableName=table_name, Item=item)
        return response


    def delete_item(self, table_name: str, partition_key: str) -> bool:
        """
        Delete item from DynamoDB table by key

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
        item_key = {
            ALL_DYANMODB_TABLES[table_name]['partition_key']: {'S': partition_key}
        }
        response = dynamodb.delete_item(TableName=table_name, Key=item_key)
        if (response['ResponseMetadata']['HTTPStatusCode'] != 200):
            raise Exception("Failed to delete item")
        return True



if __name__ == "__main__":
    print(1)
    print(DynamoDbUtils().create_item("trainspace", 
                                      {"trainspace_id": str(random.random()), "uid": "bleh", "created": datetime.now().isoformat()}))
