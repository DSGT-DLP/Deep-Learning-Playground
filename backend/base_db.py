from enum import Enum, EnumMeta
from dataclasses import dataclass
from typing import List
import boto3

class _BaseEnumMeta(EnumMeta):
    def __contains__(cls, item):
        try:
            cls(item)
        except ValueError:
            return False
        return True
    
class _BaseEnum(Enum, metaclass=_BaseEnumMeta):
    @classmethod
    def find(cls, value):
        return cls[value].name

def enumclass(cls=None, /, *, dataclass_class=None, **kwargs):  
    def process(cls):
        dataclass_fields = list(dataclass_class.__dataclass_fields__.keys())
        setattr(cls, 'Attribute', _BaseEnum('Attribute', [(field.upper(), field) for field in dataclass_fields]))
        
        for attribute in kwargs:
            if attribute in dataclass_fields:
                setattr(cls, attribute.capitalize(), _BaseEnum(attribute.capitalize(), [(element.upper(), element.upper()) for element in kwargs[attribute]]))
            else:
                del cls
                raise ValueError(f'{attribute} is not an attribute of the table')
        return cls
    
    if cls is not None or dataclass_class is None:
        raise Exception('Please provide a corresponding dataclass')
    return process

def changevar(cls=None, /, *, DataClass=None, EnumClass=None):
    def process(cls):
        cls.DataClass = DataClass
        cls.EnumClass = EnumClass
        return cls
    
    if cls is not None:
        raise Exception('Please provide the corresponding data and enum classes')
    return process


class BaseDDBUtil:
    DataClass = None
    EnumClass = None
    
    """
    Data access object for DynamoDB tables
    """    
    def __init__(self, table_name: str, region: str, partition_key: List[str], table=None):
        self.table_name = table_name
        #self.dynamodb = boto3.resource('dynamodb', region)
        self.partition_key = partition_key
        #self.table = table if table else boto3.resource('dynamodb', region).Table(self.table_name)
        
    def create_table(self, read_capacity_units: int = 10, write_capacity_units: int = 10):
        """
        Helper function to create Dynamo DB table if it doesn't exist in AWS
        """
        table = self.dynamodb.create_table(
            TableName=self.table_name,
            KeySchema=[
                {
                    'AttributeName': self.partition_key[0],
                    'KeyType': 'HASH'  # Partition key
                },
            ],
            AttributeDefinitions=[
                {
                    'AttributeName': self.partition_key[0],
                    # AttributeType defines the data type. 'S' is string type and 'N' is number type
                    'AttributeType': self.partition_key[1]
                }
            ],
            ProvisionedThroughput={
                # ReadCapacityUnits set to 10 strongly consistent reads per second
                'ReadCapacityUnits': read_capacity_units,
                'WriteCapacityUnits': write_capacity_units  # WriteCapacityUnits set to 10 writes per second
            }
        )
        self.table = table

# Usage
if __name__ == '__main__':
    @dataclass
    class StatusData:
        request_id: str
        status: str 
        timestamp: str

    @enumclass(dataclass_class=StatusData, status=['started', 'in_progress', 'success', 'failed'])
    class StatusEnums:
        pass
    
    #print(StatusEnums)
    #print(StatusEnums.Attribute)
    #print(StatusEnums.Status)
    #print(StatusEnums.Status.IN_PROGRESS.value)
    #print('IN_PROGRESS' in StatusEnums.Status)
    
    @changevar(DataClass=StatusData, EnumClass=StatusEnums)
    class StatusDDBUtil(BaseDDBUtil):
        pass
    
    BaseDDBUtil.test()
    StatusDDBUtil.test()
    a = BaseDDBUtil()
    b = StatusDDBUtil()
    a.test()
    b.test()