from dataclasses import dataclass
from backend.aws_helpers.dynamo_db_utils.base_db import BaseData, BaseDDBUtil, enumclass, changevar
from backend.common.constants import AWS_REGION, USER_TABLE_NAME

@dataclass
class UserData(BaseData):
    """Data class to hold the attribute values of a record of the user-table DynamoDB table"""
    user_id: str
    email: str
    a_uri: str
    b_uri: str
    c_uri: str
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

user_table = get_user_table()
user_table.create_record(user_id="1", email="hello@gmail.com", a_uri="a", b_uri="b", c_uri="c", timestamp="s")
print(user_table.get_record("1"))
# user_table.delete_record("1")