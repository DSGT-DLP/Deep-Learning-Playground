from enum import Enum, EnumMeta
from dataclasses import dataclass, asdict
from typing import Any, List, Dict, Literal
import boto3
from numpy import partition

class _BaseEnumMeta(EnumMeta):
    def __contains__(cls, item):
        try:
            cls(item)
        except ValueError:
            return False
        return True
    
class _BaseEnum(Enum, metaclass=_BaseEnumMeta):
    @classmethod
    def find_attribute(cls, value):
        return cls[value].name

class BaseData:
    '''Class to provide common parent type to dataclasses and for future uses'''
    pass

def enumclass(cls=None, /, *, DataClass: BaseData = None, **kwargs):  
    def process(cls):
        if not issubclass(cls, BaseData):
            raise ValueError("DataClass provided is not a dataclass")
        
        data_fields = list(DataClass.__dataclass_fields__.keys())
        setattr(cls, 'Attribute', _BaseEnum('Attribute', [(field.upper(), field) for field in data_fields]))
        
        for attribute in kwargs:
            if attribute in data_fields:
                setattr(cls, attribute.capitalize(), _BaseEnum(attribute.capitalize(), [(element.upper(), element.upper()) for element in kwargs[attribute]]))
            else:
                raise ValueError(f"{attribute} is not an attribute of the table")
        return cls
    
    if cls is not None or DataClass is None:
        raise Exception("Please provide a corresponding dataclass")
    return process

def changevar(cls=None, /, *, DataClass: BaseData = None, EnumClass: _BaseEnum = None, partition_key: List[str] = None):
    def process(cls):
        if partition_key[0] in EnumClass.Attribute:
            cls.partition_key = partition_key
        else:
            raise ValueError(f"{partition_key[0]} is not an attribute of the table")
        
        if not issubclass(DataClass, BaseData):
            raise ValueError("DataClass provided is not a subclass of type: BaseData")
        if not issubclass(EnumClass, _BaseEnum):
            raise ValueError("EnumClass provided is not a subclass of type: _BaseEnum")
        
        cls.DataClass = DataClass
        cls.EnumClass = EnumClass
        return cls
    
    if cls is not None or DataClass is None or EnumClass is None or partition_key is None:
        raise Exception("Please provide the corresponding arguments")
    return process


class BaseDDBUtil:
    DataClass: BaseData = None
    EnumClass: _BaseEnum = None
    partition_key: List[str] = None           # Stores info of the form [<attibute_name>, <attribute_type>]
    
    """
    Data access object for DynamoDB tables
    """    
    def __init__(self, table_name: str, region: str, table=None):
        self.table_name = table_name
        self.dynamodb = boto3.resource('dynamodb', region)
        self.table = table if table else boto3.resource('dynamodb', region).Table(self.table_name)
        
    def create_table(self, read_capacity_units: int = 10, write_capacity_units: int = 10) -> None:
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
        
    def create_record(self, record_data: BaseData = None, **kwargs) -> Literal['Success']:
        """
        Create a record in DynamoDB table with the data corresponding to the input parameters
        """
        self.param_checker("create", record_data=record_data, **kwargs)
        partition_key_name = self.partition_key[0]
        
        if record_data is not None:
            item = asdict(record_data)
        else:
            item = kwargs
        
        if len(item) != len(self.EnumClass.Attribute):
            raise ValueError(f"Could not create record with missing/extra attributes")   
        try:
            self.table.put_item(
                Item=item,
                ConditionExpression="attribute_not_exists({partition_key_name})"
            )
            return "Success"
        except Exception as e:
            raise ValueError(f"Could not add record. {partition_key_name} {item[partition_key_name]} already exists in the table")
        
    def get_record(self, partition_id: str) -> BaseData:
        """
        Retrieve a record with the partition_key 'partition_id' from DynamoDB table
        """
        self.param_checker("get", partition_id=partition_id)
        
        response = self.table.get_item(Key={self.partition_key[0]: partition_id})
        if 'Item' not in response:
            raise ValueError(f"Could not find a Dynamo DB item for id {id} in table {self.table_name}")
        item: Dict[int, Any] = response['Item']
        
        if len(item) != len(self.EnumClass.Attribute):
            raise ValueError(f"Could not approve record with missing/extra attributes")
        
        self.param_checker("approve", **item)
        return self.DataClass(**item)
    
    def update_record(self, partition_id: str, **kwargs) -> Literal['Success']:
        """
        Update status for a given request id
        """
        if len(kwargs) == 0:
            raise ValueError("Cannot update record without any changes")
        
        self.param_checker("update", partition_id=partition_id, **kwargs)
        partition_key_name = self.EnumClass.partition_key[0]
        expression = "SET "
        
        attribute_names = {}
        for attr in kwargs.keys():
            expression += f"{attr}=:{attr}, "
            attribute_names[f':{attr}'] = kwargs[attr]
        
        try:
            self.table.update_item(
                Key={
                    partition_key_name: partition_id
                },
                UpdateExpression=expression,
                ExpressionAttributeNames=attribute_names,
                ConditionExpression="attribute_exists(request_id)"
            )
            return "Success"
        except Exception as e:
            print(e)
            print(f"Could not update attributes for {partition_key_name} {partition_id}")
            raise ValueError(f"Could not update attributes for {partition_key_name} {partition_id}")
    
    def delete_record(self, partition_id: str) -> Literal['Success']:
        """
        Delete a record with the partition_key value 'partition_id' from DynamoDB table
        """
        self.param_checker("delete", partition_id=partition_id)
        partition_key_name = self.partition_key[0]
        
        try:
            self.table.delete_item(
                Key={
                    partition_key_name: partition_id
                },
                ConditionExpression=f"attribute_exists({partition_key_name})"
            )
            
            return "Success"
        except Exception as e:
            print(e)
            print(f"Could not delete status for {partition_key_name} {partition_id}")
            raise ValueError(f"Could not delete status for {partition_key_name} {partition_id}")
        
    def param_checker(self, operation: str, **kwargs):
        kwargs = kwargs.items()
        
        for attribute, value in kwargs:
            if attribute == 'record_data':
                if attribute is not None:
                    if type(value) is not self.DataClass:
                        raise ValueError(f"Could not {operation} record with {attribute} not of correct type")
                    if len(kwargs) > 1:
                        raise ValueError(f"Cannot provide other attributes if {attribute} is provided")
                elif len(kwargs) == 1:
                    raise ValueError(f"Could not {operation} record with missing attributes")
                pass
            elif attribute == 'partition_id':
                partition_key_name = self.partition_key[0]
                if partition_key_name in kwargs:
                    raise ValueError(f"Could provide multiple values for {partition_key_name}")
                attribute = partition_key_name
                
            if attribute not in self.EnumClass.Attribute:
                raise ValueError(f"Attribute not found in table {self.table_name}")
            elif value is None:
                raise ValueError(f"Could not {operation} record with {attribute}: None")
            elif type(value) is not self.DataClass.__dataclass_fields__[attribute].type:
                raise ValueError(f"Could not {operation} record with {attribute} not of correct type")
            elif getattr(self.EnumClass, attribute.capitalize(), None) is not None and value not in getattr(self.EnumClass, attribute.capitalize()):
                raise ValueError(f"Could not {operation} record with invalid {attribute}: {value}")
            
                

# Usage
if __name__ == '__main__':
    @dataclass
    class StatusData(BaseData):
        request_id: str
        status: str 
        timestamp: str

    @enumclass(DataClass=StatusData, partition_key=['request_id', 'S'], status=['started', 'in_progress', 'success', 'failed'])
    class StatusEnums:
        pass
    
    @changevar(DataClass=StatusData, EnumClass=StatusEnums)
    class StatusDDBUtil(BaseDDBUtil):
        pass
    
    table = StatusDDBUtil('status-table', 'us-west-2')
    print(table.get_record('234'))