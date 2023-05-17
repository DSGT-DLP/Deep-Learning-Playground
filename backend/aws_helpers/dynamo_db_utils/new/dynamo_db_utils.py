import boto3
from backend.aws_helpers.dynamo_db_utils.new.constants import ALL_DYANMODB_TABLES

dynamodb = boto3.client('dynamodb')

class DynamoDbUtils():
    def __init__(self):
        pass

    def get_item(self, table_name: str, key: str):
        if table_name not in ALL_DYANMODB_TABLES.keys():
            raise ValueError("Invalid table name: " + table_name)
        item_key = {
            ALL_DYANMODB_TABLES[table_name]['partition_key']: {'S': key}
        }
        response = dynamodb.get_item(TableName=table_name, Key=item_key)
        return response['Item']

    def create_item(self, table_name: str, key: str):
        if table_name not in ALL_DYANMODB_TABLES.keys():
            raise ValueError("Invalid table name: " + table_name)
        response = dynamodb.get_item(TableName=table_name, Key=ALL_DYANMODB_TABLES[table_name]['partition_key'])
        return response


if __name__ == "__main__":
    print(1)
    print(DynamoDbUtils().get_item("TRAINSPACE", "cd4d7451-6633-408b-a794-abeb7b99dc30"))
