from decimal import Decimal
from enum import Enum, EnumMeta
from dataclasses import asdict
from typing import Any, List, Dict, Literal
import boto3
from botocore.exceptions import ClientError

class _BaseEnumMeta(EnumMeta):
    """Defines a custom EnumMeta for _BaseEnum that supports the expression: '<attribute>' in _BaseEnum"""
    def __contains__(cls, item):
        try:
            cls(item)
        except ValueError:
            return False
        return True
    
class _BaseEnum(Enum, metaclass=_BaseEnumMeta):
    """Defines a custom Enum to provide subclasses the functionality of _BaseEnumMeta"""
    pass

class BaseData:
    """Class to provide common parent type to dataclasses, and for future uses"""
    pass

def enumclass(cls=None, /, *, DataClass: BaseData = None, **kwargs):
    """Decorator function that produces various enums based on an input table schema in the form of a BaseData.
    Enums produced include Attribute, which stores the attributes of a particular DynamoDB table, and enums associated 
    with attributes that store categorical data.

    Args:
        DataClass: dataclass of the form of a BaseData (eg: UserData)
        kwargs: attributes that store categorical data, along with the categories in list form (eg: role=["ADMIN", "USER"])
    """
    def process(cls):
        if not issubclass(DataClass, BaseData):
            raise ValueError("DataClass provided is not a dataclass")
        
        data_fields = list(DataClass.__dataclass_fields__.keys())
        setattr(cls, 'Attribute', _BaseEnum('Attribute', [(field.upper(), field) for field in data_fields]))
        
        for attribute in kwargs:
            if attribute in data_fields:
                enum_name = ''.join(map(lambda string: string.capitalize(), attribute.split("_")))
                setattr(cls, enum_name, _BaseEnum(enum_name, [(element.upper(), element.upper()) for element in kwargs[attribute]]))
            else:
                raise ValueError(f"{attribute} is not an attribute of the table")
        return cls
    
    if cls is not None or DataClass is None:
        raise Exception("Please provide a corresponding dataclass")
    return process

def changevar(cls=None, /, *, DataClass: BaseData = None, EnumClass = None, partition_key: str = None):
    """Decorator function used to assign static variables to subclasses of BaseDDBUtil that are also utilized
    in BaseDDBUtil function implementations. It is sort of analogous to how instance variables of a class are used in instance methods, 
    but the instances of the class can assign various different values to the instance variables.

    Args:
        DataClass: dataclass of the form of a BaseData (eg: UserData)
        EnumClass: class that stored enums (eg: UserEnums)
        partition_key: the partition_key of DynamoDB table
    """
    def process(cls):
        if partition_key in EnumClass.Attribute:
            cls.partition_key = partition_key
        else:
            raise ValueError(f"{partition_key} is not an attribute of the table")
        
        if not issubclass(DataClass, BaseData):
            raise ValueError("DataClass provided is not a subclass of type: BaseData")
        
        cls.DataClass = DataClass
        cls.EnumClass = EnumClass
        return cls
    
    if cls is not None or DataClass is None or EnumClass is None or partition_key is None:
        raise Exception("Please provide the corresponding arguments")
    return process

_type_mapper = {int: 'N', Decimal: 'N', str: 'S'}

class BaseDDBUtil:
    """Base class that interacts with AWS DynamoDB to manipulate information stored in the DynamoDB tables.
    Acts as a template, whose subclasses (eg: StatusDDBUtil) manipulates corresponding tables (eg: status-table)"""
    DataClass: BaseData = None
    EnumClass = None
    partition_key: str = None               # Stores partition_key of the table
    
    def __init__(self, table_name: str, region: str):
        self.table_name = table_name
        self.dynamodb = boto3.resource('dynamodb', region)
        try:
            self.table = self.dynamodb.Table(self.table_name)
            self.table.table_id
        except Exception:
            self.create_table()          
        
    def create_table(self, read_capacity_units: int = 10, write_capacity_units: int = 10) -> None:
        """Function to create a DynamoDB table based on instance fields if it does not exist in AWS"""
        
        table = self.dynamodb.create_table(
            TableName=self.table_name,
            KeySchema=[
                {
                    'AttributeName': self.partition_key,
                    'KeyType': 'HASH'  # Partition key
                },
            ],
            AttributeDefinitions=[
                {
                    'AttributeName': self.partition_key,
                    # AttributeType defines the data type. 'S' is string type and 'N' is number type
                    'AttributeType': _type_mapper[self.DataClass.__dataclass_fields__[self.partition_key].type]
                }
            ],
            ProvisionedThroughput={
                # ReadCapacityUnits set to 10 strongly consistent reads per second
                'ReadCapacityUnits': read_capacity_units,
                'WriteCapacityUnits': write_capacity_units  # WriteCapacityUnits set to 10 writes per second
            }
        )
        self.table = table
        
    def create_gsi(self, attribute_name: str, attribute_type: str = "HASH", projection_type: str = "KEYS_ONLY", 
                   read_capacity: int = 10, write_capacity: int = 10, nonkey_attributes: List[str] = None) -> Literal['Success']:
        """Function that adds a global secondary index to the associated table"""
        
        if (type(attribute_name) != str or type(attribute_type) != str or type(read_capacity) != int or 
                type(write_capacity) != int or type(projection_type) != str):
            raise ValueError("Cannot create global secondary index with invalid argument")        
        
        if attribute_name not in self.EnumClass.Attribute:
            raise ValueError(f"Attribute '{attribute_name}' not found in table {self.table_name}")
        if attribute_type not in ['HASH', 'RANGE']:
            raise ValueError(f"Invalid attribute_type argument")
        if projection_type not in ['ALL', 'KEYS_ONLY', 'INCLUDE']:
            raise ValueError(f"Invalid projection_type argument")
        
        if projection_type == 'INCLUDE':
            if nonkey_attributes is None:
                raise ValueError(f"nonkey_attributes need to be provided for projection_type 'INCLUDE'")
            else:
                for attribute in nonkey_attributes:
                    if attribute not in self.EnumClass.Attribute:
                        raise ValueError(f"Attribute '{attribute}' not found in table {self.table_name}")
        elif nonkey_attributes is not None:
            raise ValueError(f"nonkey_attributes should not be provided for projection_type 'f{projection_type}'")
    
        projection = {'ProjectionType': projection_type}
        if projection_type == 'INCLUDE':
            projection['NonKeyAttributes'] = nonkey_attributes
        
        self.table = self.table.update(
            AttributeDefinitions=[
                {
                    'AttributeName': attribute_name,
                    'AttributeType': _type_mapper[self.DataClass.__dataclass_fields__[attribute_name].type]
                }
            ],
            GlobalSecondaryIndexUpdates=[
                {
                    'Create': {
                        'IndexName': attribute_name,
                        'KeySchema': [
                            {
                                'AttributeName': attribute_name,
                                'KeyType': attribute_type
                            },
                        ],
                        'Projection': projection,
                        'ProvisionedThroughput': {
                            'ReadCapacityUnits': read_capacity,
                            'WriteCapacityUnits': write_capacity
                        }
                    }
                }
            ]
        )
        return "Success"
        
    def create_record(self, record_data: BaseData = None, **kwargs) -> Literal['Success']:
        """Function to create a record in the associated DynamoDB table with the data corresponding to the input parameters.
        Either takes a BaseData input, or the attribute values as keyword arguments"""
        
        self.__param_checker("create", record_data=record_data, **kwargs)
        partition_key_name = self.partition_key
        
        if record_data is not None:
            item = asdict(record_data)
        else:
            item = kwargs
        
        if len(item) != len(self.EnumClass.Attribute):
            raise ValueError(f"Could not create record with missing/extra attributes")   
        try:
            self.table.put_item(
                Item=item,
                # ConditionExpression=f"attribute_not_exists({partition_key_name})"
            )
            return "Success"
        except ClientError as e:
            if e.response['Error']['Code'] == 'ConditionalCheckFailedException':
                raise ValueError(f"Could not add record. {partition_key_name} {item[partition_key_name]} already exists in the table")
            else:
                raise e
        
    def get_record(self, partition_id: Any) -> BaseData:
        """Function to retrieve a record with the partition_key values as attribute 'partition_id' from 
            the associated DynamoDB table"""
        
        self.__param_checker("get", partition_id=partition_id)
        
        response = self.table.get_item(Key={self.partition_key: partition_id})
        if 'Item' not in response:
            raise ValueError(f"Could not find a DynamoDB item for {self.partition_key} {partition_id} in table {self.table_name}")
        item: Dict[str, Any] = response['Item']
        
        if len(item) != len(self.EnumClass.Attribute):
            raise ValueError(f"Could not approve record with missing/extra attributes")
        
        item = self.__number_decoder(item)        
        self.__param_checker("approve", **item)
        return self.DataClass(**item)
        
    def update_record(self, partition_id: Any, **kwargs) -> Literal['Success']:
        """Function to update a record with the partition_key values as attribute 'partition_id' from 
            the associated DynamoDB table. It takes in changes in attributes as keyword arguments"""
        if len(kwargs) == 0:
            raise ValueError("Cannot update record without any changes")
        
        self.__param_checker("update", partition_id=partition_id, **kwargs)
        partition_key_name = self.partition_key
        
        expression = []
        attribute_names = {}
        attribute_values = {}
        for attr in kwargs.keys():
            expression.append(f"#{attr}=:{attr}")
            attribute_names[f'#{attr}'] = attr
            attribute_values[f':{attr}'] = kwargs[attr]
        expression = f"SET {', '.join(expression)}"
        
        try:
            self.table.update_item(
                Key={
                    partition_key_name: partition_id
                },
                UpdateExpression=expression,
                ExpressionAttributeNames=attribute_names,
                ExpressionAttributeValues=attribute_values,
                ConditionExpression=f"attribute_exists({partition_key_name})"
            )
            return "Success"
        except ClientError as e:
            if e.response['Error']['Code'] == 'ConditionalCheckFailedException':
                raise ValueError(f"Could not find a DynamoDB item to update for {self.partition_key} {partition_id} in table {self.table_name}")
            else:
                raise e
    
    def delete_record(self, partition_id: Any) -> Literal['Success']:
        """Function to delete a record with the partition_key values as attribute 'partition_id' from 
            the associated DynamoDB table"""
            
        self.__param_checker("delete", partition_id=partition_id)
        partition_key_name = self.partition_key
        
        try:
            self.table.delete_item(
                Key={
                    partition_key_name: partition_id
                },
                ConditionExpression=f"attribute_exists({partition_key_name})"
            )
            return "Success"
        except ClientError as e:
            if e.response['Error']['Code'] == 'ConditionalCheckFailedException':
                raise ValueError(f"Could not find a DynamoDB item to delete for {self.partition_key} {partition_id} in table {self.table_name}")
            else:
                raise e
        
    def __param_checker(self, operation: str, **kwargs):
        """Helper function to perform a validity check on the parameters of the BaseDDBUtil functions"""
        
        kwargs_items = kwargs.items()
        for attribute, value in kwargs_items:
            if attribute == 'record_data':
                if value is not None:
                    if type(value) is not self.DataClass:
                        raise ValueError(f"Could not {operation} record with {attribute} not of correct type")
                    if len(kwargs) > 1:
                        raise ValueError(f"Cannot provide other attributes if {attribute} is provided")
                    self.__param_checker(operation, **asdict(value))
                elif len(kwargs) == 1:
                    raise ValueError(f"Could not {operation} record with missing attributes")
                continue
            elif attribute == 'partition_id':
                partition_key_name = self.partition_key
                if partition_key_name in kwargs:
                    raise ValueError(f"Cannot {operation} record with multiple values for {partition_key_name}")
                attribute = partition_key_name
                
            if attribute not in self.EnumClass.Attribute:
                raise ValueError(f"Attribute '{attribute}' not found in table {self.table_name}")
            elif value is None:
                pass
            elif type(value) is not self.DataClass.__dataclass_fields__[attribute].type:
                raise ValueError(f"Could not {operation} record with {attribute} not of correct type")
            elif getattr(self.EnumClass, attribute.capitalize(), None) is not None and value not in getattr(self.EnumClass, attribute.capitalize()):
                raise ValueError(f"Could not {operation} record with invalid {attribute}: {value}")
            
    def __number_decoder(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Helper function that converts any Decimal attributes into their associated types according to BaseData schema. 
            To be used on records retrieved from DynamoDB tables."""
            
        for key, value in item.items():
            if type(value) is Decimal:
                if key in self.EnumClass.Attribute:
                    key_type = self.DataClass.__dataclass_fields__[key].type
                    if type(value) != key_type:
                        if value != key_type(value):
                            raise ValueError(f"Could not approve record with {key} not of correct type")
                        item[key] = key_type(value)
                else:
                    raise ValueError(f"Attribute {key} not found in table {self.table_name}")
            pass
        return item