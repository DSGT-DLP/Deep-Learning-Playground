ALL_DYANMODB_TABLES = {
    "trainspace": {
        "partition_key": "trainspace_id",
        "gsi": "uid"
    },
    "dlp-file-upload-table": {
        "partition_key": "s3_uri",
        "gsi": "uid"
    }
}
