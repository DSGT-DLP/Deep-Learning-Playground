from dataclasses import dataclass
from backend.aws_helpers.dynamo_db_utils.base_db import (
    BaseData,
    BaseDDBUtil,
    enumclass,
    changevar,
)
from backend.common.constants import EXECUTION_TABLE_NAME, AWS_REGION


@dataclass
class UserProgressData(BaseData):
    uid: str
    progressData: str


@enumclass(DataClass=UserProgressData)
class UserProgressDataEnums:
    pass


@changevar(
    DataClass=UserProgressData, EnumClass=UserProgressDataEnums, partition_key="uid"
)
class UserProgressDDBUtil(BaseDDBUtil):
    pass
