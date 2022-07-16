from dataclasses import dataclass
from base_db import BaseData, BaseDDBUtil, enumclass, changevar
from constants import USER_TABLE_NAME, AWS_REGION

@dataclass
class UserData(BaseData):
    request_id: str
    status: str 
    timestamp: str
    
@enumclass(DataClass=UserData, status=['started', 'in_progress', 'success', 'failed'])
class UserEnums:
    pass

@changevar(DataClass=UserData, EnumClass=UserEnums, partition_key=['request_id', 'S'], )
class UserDDBUtil(BaseDDBUtil):
    pass

def get_status_table(table_name:str = USER_TABLE_NAME, region:str = AWS_REGION) -> UserDDBUtil:
    """
    Retrieves a user table as an instance of UserDDBUtil
    """
    return UserDDBUtil(table_name, region)

print('hi')
table = get_status_table()
print(table)
print(table.get_record('123'))