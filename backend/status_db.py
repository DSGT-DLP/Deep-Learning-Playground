from typing import Dict, Any
from enum import Enum, EnumMeta
from dataclasses import dataclass, asdict 
import boto3
from backend.constants import STATUS_TABLE_NAME, AWS_REGION

class StatusEnumMeta(EnumMeta):
    def __contains__(cls, item):
        try:
            cls(item)
        except ValueError:
            return False
        return True 
    

class StatusAttribute(Enum, metaclass=StatusEnumMeta):
    """
    Enum class to represent all valid attributes for a Dynamo DB
    item from the Status table
    """
    REQUEST_ID = "request_id" #UUID/GUID for this request
    STATUS = "status" #status of request (started, in progress, success, fail)
    TIMESTAMP = "timestamp"

class StatusEnum(Enum):
    STARTED = 'STARTED'
    IN_PROGRESS = 'IN_PROGRESS'
    SUCCESS = 'SUCCESS'
    FAILED = 'FAILED'

@dataclass
class StatusData:
    """
    Data class to represent a Dynamo DB item from the status table
    """
    request_id: str
    status: str 
    timestamp: str

class StatusDDBUtil:
    """
    Data access object for Status table
    """    
    def __init__(self, table_name: str, region: str, table=None):
        self.table_name = table_name
        self.dynamodb = boto3.resource('dynamodb', region)
        self.table = table if table else boto3.resource('dynamodb', region).Table(self.table_name)
    
    def create_table(self):
        """
        Helper function to create Dynamo DB table if it doesn't exist in AWS
        """
        table = self.dynamodb.create_table(
            TableName=self.table_name,
            KeySchema=[
                {
                    'AttributeName': 'request_id',
                    'KeyType': 'HASH'  # Partition key
                },
            ],
            AttributeDefinitions=[
                {
                    'AttributeName': 'request_id',
                    # AttributeType defines the data type. 'S' is string type and 'N' is number type
                    'AttributeType': 'S'
                }
            ],
            ProvisionedThroughput={
                # ReadCapacityUnits set to 10 strongly consistent reads per second
                'ReadCapacityUnits': 10,
                'WriteCapacityUnits': 10  # WriteCapacityUnits set to 10 writes per second
            }
            
        )
        self.table = table 
        
    def create_status_entry(self, data: StatusData):
        item = {k: v for k, v in asdict(data).items() if v is not None}
        
        request_id = item.get('request_id')
        if request_id is None:
            raise ValueError(f"Could not add record with request_id: None")
        elif type(request_id) is not str:
            raise ValueError(f"Could not add record with request_id not of type 'str'")
        elif item.get('status') not in set(member.value for member in StatusEnum):
            raise ValueError(f"Could not add record with invalid status: {item.get('status')}")
        elif item.get('timestamp') is None:
            raise ValueError(f"Could not add record with missing timestamp")

        try:
            self.table.put_item(
                Item=item,
                ConditionExpression="attribute_not_exists(request_id)"
            )
            return "Success"
        except Exception as e:
            raise ValueError(f"Could not add record. request_id {request_id} already exists in the table")    
        
    def get_status_entry(self, request_id: str) -> StatusData:
        """
        Retrieve info regarding specific request id from status table
        """
        if request_id is None:
            raise ValueError(f"Invalid request_id None not allowed")
        elif type(request_id) is not str:
            raise ValueError(f"Could not add record with request_id not of type 'str'")
            
        response = self.table.get_item(Key={StatusAttribute.REQUEST_ID.value: request_id})
        
        if 'Item' not in response:
            raise ValueError(
                f"Could not find a Dynamo DB item for id {id} in table {self.table_name}"
            )
        
        item: Dict[int, Any] = response['Item']
        for attribute in item:
            if attribute not in StatusAttribute:
                raise ValueError(
                    f"Found invalid attribute {attribute} for id {request_id} in table {self.table_name}"
                )
        
        return StatusData(
            request_id=set_status_data(item, StatusAttribute.REQUEST_ID),
            status=set_status_data(item, StatusAttribute.STATUS),
            timestamp=set_status_data(item, StatusAttribute.TIMESTAMP)
        )
        
    def update_status_entry(self, request_id: str, new_status: StatusEnum):
        """
        Update status for a given request id
        """
        if request_id is None:
            raise ValueError(f"Could not add record with request_id: None")
        elif type(request_id) is not str:
            raise ValueError(f"Could not add record with request_id not of type 'str'")
        elif new_status not in set(member.value for member in StatusEnum):
            raise ValueError(f"Could not add record with invalid status: {new_status}")
        
        try:
            self.table.update_item(
                Key={
                    'request_id': request_id
                },
                UpdateExpression="set #s=:status",
                ExpressionAttributeValues={
                    ":status": new_status
                },
                ExpressionAttributeNames = {
                    "#s": "status"
                },
                ConditionExpression="attribute_exists(request_id)"
            )
            return "Success"
        except Exception as e:
            print(e)
            print(f"Oops. Could not update status for request id {request_id}")
            raise ValueError(f"Oops. Could not update status to {new_status} for request id {request_id}")
    
    def delete_status_entry(self, request_id: str):
        """
        Delte status for a given request id
        """
        if request_id is None:
            raise ValueError(f"Could not add record with request_id: None")
        elif type(request_id) is not str:
            raise ValueError(f"Could not add record with request_id not of type 'str'")
        
        try:
            self.table.delete_item(
                Key={
                    'request_id': request_id
                },
                ConditionExpression="attribute_exists(request_id)"
            )
            
            return "Success"
        except Exception as e:
            print(e)
            print(f"Oops. Could not delete status for request id {request_id}")
            raise ValueError(f"Oops. Could not delete status for request id {request_id}")

def set_status_data(item: Dict[str, Any], attribute: StatusAttribute):
    """
    Set status data attribute from Dynamo DB item if attribute is present
    """
    return item[attribute.value] if attribute.value in item else None

def get_status_table(table_name:str = STATUS_TABLE_NAME, region:str = AWS_REGION) -> StatusDDBUtil:
    """
    Retrieves a status table as an instance of StatusDDBUtil
    """
    return StatusDDBUtil(table_name, region)