from dataclasses import dataclass
from backend.aws_helpers.dynamo_db_utils.base_db import BaseData, BaseDDBUtil, enumclass, changevar
from backend.common.constants import AWS_REGION, USER_TABLE_NAME

@dataclass
class UserData(BaseData):
    """Data class to hold the attribute values of a record of the user-table DynamoDB table"""
    user_id: str
    email: str
    result_uri: str
    onnx_uri: str
    pt_uri: str
    timestamp: str
    
@enumclass(DataClass=UserData)
class UserEnums:
    """Enum class with no values specified but needed to run"""
    pass

@changevar(DataClass=UserData, EnumClass=UserEnums, partition_key='user_id')
class UserDDBUtil(BaseDDBUtil):
    """Class that interacts with AWS DynamoDB to manipulate information stored in the user-table DynamoDB table"""
    pass

def get_user_table(region:str = AWS_REGION) -> BaseDDBUtil:
    """Retrieves the user-table of an input region as an instance of StatusDDBUtil"""
    return UserDDBUtil(USER_TABLE_NAME, region)