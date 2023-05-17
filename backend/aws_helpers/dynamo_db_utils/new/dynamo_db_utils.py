import boto3
from backend.aws_helpers.dynamo_db_utils.new.constants import ALL_DYANMODB_TABLES

dynamodb = boto3.client('dynamodb')

class DynamoDbUtils():
    def __init__(self):
        pass

    def get_item(self, table_name: str, key: str) -> dict:
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
        if table_name not in ALL_DYANMODB_TABLES.keys():
            raise ValueError("Invalid table name: " + table_name)
        item_key = {
            ALL_DYANMODB_TABLES[table_name]['partition_key']: {'S': key}
        }
        response = dynamodb.get_item(TableName=table_name, Key=item_key)
        if response.get('Item') is None:
            raise Exception("Item not found")
        
        return response['Item']


    def delete_item(self, table_name: str, key: str) -> dict:
        """
        Delete item from DynamoDB table by key

        Args:
            table_name (str): Name of DynamoDB table
            key (str): Key of item to delete

        Returns:
            dict: Response from DynamoDB delete_item call with 200 status code

        Raises:
            ValueError: If table_name is not a valid table name
            Exception: If DynamoDB delete_item call fails for any reason
        """
        if table_name not in ALL_DYANMODB_TABLES.keys():
            raise ValueError("Invalid table name: " + table_name)
        item_key = {
            ALL_DYANMODB_TABLES[table_name]['partition_key']: {'S': key}
        }
        response = dynamodb.delete_item(TableName=table_name, Key=item_key)
        if (response['ResponseMetadata']['HTTPStatusCode'] != 200):
            raise Exception("Failed to delete item")
        return response



if __name__ == "__main__":
    print(1)
    print(DynamoDbUtils().get_item("trainspace", "cd4d7451-6633-408b-a794-abeb7b99dc30"))
