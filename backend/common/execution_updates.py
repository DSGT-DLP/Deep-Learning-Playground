import datetime
import threading

from backend.aws_helpers.dynamo_db_utils.execution_db import get_execution_table, ExecutionData

execution_dict = {}

def get_execution_dict():
    return execution_dict

def regular_updates(execution_id):
    execution_db = get_execution_table()
    def update():
        if execution_id in execution_dict and execution_dict[execution_id] != 100:
            threading.Timer(30, update).start()
            execution_db.update_record(execution_id, progress=execution_dict[execution_id])
    update()

def create_execution(execution_id, user_id, name, data_source):
    execution_db = get_execution_table()
    record = ExecutionData(execution_id, user_id, name, datetime.datetime.now().isoformat(), data_source, 'STARTING', 0)
    execution_db.create_record(record)

def initialize_training(execution_id):
    execution_db = get_execution_table()
    execution_db.update_record(execution_id, status='TRAINING')
    execution_dict[execution_id] = 0

def end_training(execution_id, success: bool):
    execution_db = get_execution_table()
    status = 'SUCCESS' if success else 'ERROR'
    execution_db.update_record(execution_id, status=status, progress=100)
    execution_dict.pop(execution_id)

def end_training_success(execution_id):
    end_training(execution_id, True)
    
def end_training_failure(execution_id):
    end_training(execution_id, False)