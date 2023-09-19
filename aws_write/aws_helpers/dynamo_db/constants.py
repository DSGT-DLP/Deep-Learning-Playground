from enum import Enum


TRAINSPACE_TABLE_NAME = "trainspace"

ALL_DYANMODB_TABLES = {
    TRAINSPACE_TABLE_NAME: {"partition_key": "trainspace_id", "gsi": "uid"}
}

TrainStatus = Enum(
    "TrainStatus", ["QUEUED", "STARTING", "UPLOADING", "TRAINING", "SUCCESS", "ERROR"]
)
