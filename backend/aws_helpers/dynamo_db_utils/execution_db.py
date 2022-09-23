from dataclasses import dataclass
from backend.aws_helpers.dynamo_db_utils.base_db import BaseData, BaseDDBUtil, enumclass, changevar
from backend.common.constants import EXECUTION_TABLE_NAME, AWS_REGION

@dataclass
class ExecutionData(BaseData):
    """Data class to hold the attribute values of a record of the execution-table DynamoDB table"""
    execution_id: str
    user_id: str
    name: str
    timestamp: str
    data_source: str
    status: str
    progress: int
    
@enumclass(
    DataClass=ExecutionData,
    data_source=['TABULAR', 'PRETRAINED', 'IMAGE', 'AUDIO', 'TEXTUAL'],
    status=['QUEUED', 'STARTING', 'UPLOADING', 'TRAINING', 'SUCCESS', 'ERROR']
)
class ExecutionEnums:
    """Class that holds the enums associated with the ExecutionDDBUtil class. It includes:
        ExecutionEnums.Attribute - Enum that defines the schema of the execution-table. It holds the attribute names of the table
        ExecutionEnums.Execution_Source - Enum that defines the categorical values associated with the 'execution_source' attribute"""
    pass

@changevar(DataClass=ExecutionData, EnumClass=ExecutionEnums, partition_key='execution_id')
class ExecutionDDBUtil(BaseDDBUtil):
    """Class that interacts with AWS DynamoDB to manipulate information stored in the execution-table DynamoDB table"""
    pass

def get_execution_table(region:str = AWS_REGION) -> BaseDDBUtil:
    """Retrieves the execution-table of an input region as an instance of ExecutionDDBUtil"""
    return ExecutionDDBUtil(EXECUTION_TABLE_NAME, region)