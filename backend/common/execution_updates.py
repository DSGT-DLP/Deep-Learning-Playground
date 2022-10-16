from flask import request
import datetime
import threading
import jwt
import json

from backend.aws_helpers.aws_secrets_utils.aws_secrets import get_secret
from backend.aws_helpers.dynamo_db_utils.execution_db import get_execution_table, ExecutionData

execution_db = get_execution_table()
execution_dict = {}
encryption_key = json.loads(get_secret("all/encryption"))["execution_id_encryption"]

def get_execution_dict():
    return execution_dict

def create_execution_id(user_id):
    return jwt.encode({"user_id": user_id, "timestamp": datetime.datetime.now().isoformat()}, encryption_key)

def get_records():
    return execution_db.get_user_records()

def regular_updates(execution_id):
    def update():
        if execution_id in execution_dict and execution_dict[execution_id] != 100:
            threading.Timer(30, update).start()
            execution_db.update_record(execution_id, progress=execution_dict[execution_id])
    update()

def create_execution(name, file_name, data_source, isUpload):
    user_id = request.environ["user"]["user_id"]
    execution_id = create_execution_id(user_id)
    
    status = "UPLOADING" if isUpload else "STARTING"
    record = ExecutionData(execution_id, user_id, name, file_name, datetime.datetime.now().isoformat(), data_source, status, 0)
    execution_db.create_record(record)
    
    
    return execution_id

def upload_to_start(execution_id):
    execution_db.update_record(execution_id, status="STARTING")

def initialize_training(execution_id):
    execution_db.update_record(execution_id, status="TRAINING")
    execution_dict[execution_id] = 0

def end_training_success(execution_id):
    execution_db.update_record(execution_id, status="SUCCESS", progress=100)
    execution_dict.pop(execution_id)
    
def end_training_failure(execution_id):
    execution_db.update_record(execution_id, status="ERROR")
    if execution_id in execution_dict:
        execution_dict.pop(execution_id)