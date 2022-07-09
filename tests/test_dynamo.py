import pytest
#from backend import email_notifier as em
import json
from backend.status_db import *
import datetime

"""
status_ddb = get_status_table("us-west-2")
dummy_status = {"request_id": "dummy", "status": StatusEnum.STARTED.name, "timestamp": datetime.datetime.now().isoformat()}
status_data = StatusData(set_status_data(dummy_status, StatusAttribute.REQUEST_ID),
                            set_status_data(dummy_status, StatusAttribute.STATUS),
                            set_status_data(dummy_status, StatusAttribute.TIMESTAMP))
status_ddb.create_status_entry(status_data)
status_ddb.update_status("dummy", StatusEnum.SUCCESS.name)
status_ddb.delete_status("dummy",StatusEnum.SUCCESS.name)

"""

@pytest.fixture
def teardown():
    '''
    Deletes any records added by the test cases.
    '''
    test_records = []             # list containing ids of records added in tests
    
    yield test_records
    
    if (len(test_records) != 0):
        status_ddb = get_status_table("us-west-2")
        for id in test_records:
            try:
                status_ddb.delete_status(id, StatusEnum.SUCCESS.name)
            except:
                pass

@pytest.mark.parametrize(
    "id,status,timestamp",
    [
        (
            "1",
            StatusEnum.STARTED.name,
            datetime.datetime.now().isoformat()
        ),
        (
            "2",
            StatusEnum.IN_PROGRESS.name,
            datetime.datetime.now().isoformat()
        ),
        (
            "3",
            StatusEnum.SUCCESS.name,
            datetime.datetime.now().isoformat()
        ),
        (
            "4",
            StatusEnum.FAILED.name,
            datetime.datetime.now().isoformat()
        )        
    ],
)
def test_status_entry(id,status,timestamp, teardown):
    status_ddb = get_status_table("us-west-2")
    dummy_status = {"request_id": id, "status": status, "timestamp": timestamp}
    status_data = StatusData(set_status_data(dummy_status, StatusAttribute.REQUEST_ID),
                                set_status_data(dummy_status, StatusAttribute.STATUS),
                                set_status_data(dummy_status, StatusAttribute.TIMESTAMP))
    teardown.append(id)
    output = status_ddb.create_status_entry(status_data)
    assert output == "Success"

@pytest.mark.parametrize(
    "id,status,timestamp",
    [
        (
            None,
            StatusEnum.SUCCESS.name,
            datetime.datetime.now().isoformat()
        ),
        (
            "2",
            None,
            datetime.datetime.now().isoformat()
        ),
        (
            "3",
            StatusEnum.SUCCESS.name,
            None
        ),
        (
            "dummy",
            StatusEnum.SUCCESS.name,
            datetime.datetime.now().isoformat()
        ),
        (
            5,
            StatusEnum.SUCCESS.name,
            datetime.datetime.now().isoformat()
        ),
        (
            "6",
            "random status string",
            datetime.datetime.now().isoformat()
        )
    ]
)
def test_status_entry_invalid_param(id,status,timestamp, teardown):
    status_ddb = get_status_table("us-west-2")
    dummy_status = {"request_id": id, "status": status, "timestamp": timestamp}
    status_data = StatusData(set_status_data(dummy_status, StatusAttribute.REQUEST_ID),
                                set_status_data(dummy_status, StatusAttribute.STATUS),
                                set_status_data(dummy_status, StatusAttribute.TIMESTAMP))
    with pytest.raises(ValueError) as err:
        teardown.append(id)
        status_ddb.create_status_entry(status_data)
    
 
@pytest.mark.parametrize(
    "id,status,time_stamp",
    [
        (
            "1",
            StatusEnum.STARTED.name,
            datetime.datetime.now().isoformat()
        ),
    ],
)    
def test_status_entry_duplicates(id, status, time_stamp, teardown):
    status_ddb = get_status_table("us-west-2")
    dummy_status1 = {"request_id": id, "status": status, "timestamp": time_stamp}
    status_data1 = StatusData(set_status_data(dummy_status1, StatusAttribute.REQUEST_ID),
                                set_status_data(dummy_status1, StatusAttribute.STATUS),
                                set_status_data(dummy_status1, StatusAttribute.TIMESTAMP))
    teardown.append(id)
    output1 = status_ddb.create_status_entry(status_data1)
    assert output1 == "Success"
    
    dummy_status2 = {"request_id": id, "status": status, "timestamp": datetime.datetime.now().isoformat()}
    status_data2 = StatusData(set_status_data(dummy_status2, StatusAttribute.REQUEST_ID),
                                set_status_data(dummy_status2, StatusAttribute.STATUS),
                                set_status_data(dummy_status2, StatusAttribute.TIMESTAMP))
    
    
    with pytest.raises(ValueError) as err:
        # If request_id already exists, it should not be altered by create_status_entry
        status_ddb.create_status_entry(status_data2)
    status_ddb.delete_status(id, StatusEnum.SUCCESS.name)