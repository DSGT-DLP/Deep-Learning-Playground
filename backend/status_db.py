from dataclasses import dataclass
from base_db import BaseData, BaseDDBUtil, enumclass, changevar
from constants import STATUS_TABLE_NAME, AWS_REGION
import datetime

@dataclass
class StatusData(BaseData):
    """Data class to hold the attribute values of a record of the status-table DynamoDB table"""
    request_id: str
    status: str
    timestamp: str
    
@enumclass(DataClass=StatusData, status=['STARTED', 'IN_PROGRESS', 'SUCCESS', 'FAILED'])
class StatusEnums:
    """Class that holds the enums associated with the StatusDDBUtil class. It includes:
        StatusEnums.Attribute - Enum that defines the schema of the status-table. It holds the attribute names of the table
        StatusEnums.Status - Enum that defines the categorical values associated with the 'status' attribute"""
    pass

@changevar(DataClass=StatusData, EnumClass=StatusEnums, partition_key='request_id')
class StatusDDBUtil(BaseDDBUtil):
    """Class that interacts with AWS DynamoDB to manipulate information stored in the status-table DynamoDB table"""
    pass

def get_status_table(region:str = AWS_REGION) -> BaseDDBUtil:
    """Retrieves the status-table of an input region as an instance of StatusDDBUtil"""
    return StatusDDBUtil(STATUS_TABLE_NAME, region)