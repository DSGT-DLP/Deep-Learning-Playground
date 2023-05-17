ALL_DYANMODB_TABLES = {
    "TRAINSPACE": {
        "partition_key": "uid",
        "gsi": None
    },
    "DLP-FILE-UPLOAD-TABLE": {
        "partition_key": "s3_uri",
        "gsi": "uid"
    }
}
