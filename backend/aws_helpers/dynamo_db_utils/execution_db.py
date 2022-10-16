from backend.aws_helpers.dynamo_db_utils.base_db import BaseData, BaseDDBUtil, enumclass, changevar
from backend.common.constants import EXECUTION_TABLE_NAME, AWS_REGION, MONTHS
from boto3.dynamodb.conditions import Key
from dataclasses import dataclass
from typing import List

@dataclass
class ExecutionData(BaseData):
    """Data class to hold the attribute values of a record of the execution-table DynamoDB table"""
    execution_id: str
    user_id: str
    name: str
    file_name: str
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

@changevar(DataClass=ExecutionData, EnumClass=ExecutionEnums, partition_key='execution_id')
class ExecutionDDBUtil(BaseDDBUtil):
    """Class that interacts with AWS DynamoDB to manipulate information stored in the execution-table DynamoDB table"""
    
    def get_user_records(self, user_id: str) -> List[ExecutionData]:
        """Function to grab the records corresponding to a user_id"""
        query = self.table.query(
            IndexName="user_id",
            Select="ALL_ATTRIBUTES",
            KeyConditionExpression=Key("user_id").eq(user_id)
        )["Items"]
        
        user_records = []
        for record in query:
            record = self.number_decoder(record)
            record.pop("user_id")
            record["status"] = record["status"].capitalize()
            record["data_source"] = record["data_source"].capitalize()
            
            date = record["timestamp"][:10].split("-")
            record["timestamp"] = f"{MONTHS[date[1]]} {date[2]}, {date[0]}"
            user_records.append(record)
            
        return user_records

def get_execution_table(region:str = AWS_REGION) -> BaseDDBUtil:
    """Retrieves the execution-table of an input region as an instance of ExecutionDDBUtil"""
    return ExecutionDDBUtil(EXECUTION_TABLE_NAME, region)