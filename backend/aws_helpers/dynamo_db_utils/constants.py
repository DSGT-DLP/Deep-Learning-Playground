EXECUTION_TABLE_NAME = "execution-table"
USERPROGRESS_TABLE_NAME = "userprogress_table"
TRAINSPACE_TABLE_NAME = "trainspace"

ALL_DYANMODB_TABLES = {
    TRAINSPACE_TABLE_NAME: {"partition_key": "trainspace_id", "gsi": "uid"},
    EXECUTION_TABLE_NAME: {"partition_key": "execution_id", "gsi": "user_id"},
    USERPROGRESS_TABLE_NAME: {
        "partition_key": "uid",
    },
}
