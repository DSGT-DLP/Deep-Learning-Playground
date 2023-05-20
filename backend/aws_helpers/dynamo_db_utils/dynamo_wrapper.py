import boto3
from botocore.exceptions import ClientError


class DynamoDB:
    def __init__(self, table_name: str, region: str):
        self.table_name = table_name
        self.dynamodb = boto3.resource("dynamodb", region)

    def create_table(self, attribute_definitions, key_schema, provisioned_throughput):
        try:
            response = self.dynamodb.create_table(
                TableName=self.table_name,
                AttributeDefinitions=attribute_definitions,
                KeySchema=key_schema,
                ProvisionedThroughput=provisioned_throughput,
            )
            return response
        except ClientError as e:
            # Handle any errors that occurred during the create_table operation
            print(e)
            return None

    def put_item(self, item):
        try:
            response = self.dynamodb.put_item(TableName=self.table_name, Item=item)
            return response
        except ClientError as e:
            # Handle any errors that occurred during the put_item operation
            print(e)
            return None

    def get_item(self, key):
        try:
            response = self.dynamodb.get_item(TableName=self.table_name, Key=key)
            return response["Item"]
        except ClientError as e:
            # Handle any errors that occurred during the get_item operation
            print(e)
            return None

    def update_item(self, key, update_expression, expression_attribute_values):
        try:
            response = self.dynamodb.update_item(
                TableName=self.table_name,
                Key=key,
                UpdateExpression=update_expression,
                ExpressionAttributeValues=expression_attribute_values,
            )
            return response
        except ClientError as e:
            # Handle any errors that occurred during the update_item operation
            print(e)
            return None
