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

# To run a particular test case and monitor its effect in the aws console, comment out any "teardown.append(id)"
# and run the test case through the following command: pytesty tests/test_dynamo.py -k "<test_case>"


def create_status_data(id: str, status: str, timestamp: str, teardown: list[str]) -> StatusData:
    '''
    Helper method to create a StatuData object from input params
    '''
    dummy_status = {"request_id": id, "status": status, "timestamp": timestamp}
    status_data = StatusData(set_status_data(dummy_status, StatusAttribute.REQUEST_ID),
                                set_status_data(dummy_status, StatusAttribute.STATUS),
                                set_status_data(dummy_status, StatusAttribute.TIMESTAMP))
    teardown.append(id)
    return status_data

def valid_status_entry_helper(teardown: list[str], status_ddb: StatusDDBUtil) -> None:
    """
    Helper method to temporarily add records in DynamoDB table to test get_record, update_status, and delete_status functions
    """
    records = [
        ("0", StatusEnum.STARTED.name, datetime.datetime.now().isoformat()),
        ("1", StatusEnum.IN_PROGRESS.name, datetime.datetime.now().isoformat()),
        ("2", StatusEnum.FAILED.name, datetime.datetime.now().isoformat())
    ]
    
    all_status_data = []
    for record in records:
        status_data = create_status_data(record[0], record[1], record[2], teardown)
        all_status_data.append(status_data)
        status_ddb.create_status_entry(status_data)
    return all_status_data

@pytest.fixture
def teardown() -> None:
    '''
    Deletes any records added by the test cases.
    '''
    #initializing test
    test_records = []             # list containing ids of records added in tests
    
    #performing test
    yield test_records 
    
    #tearing down tests
    if (len(test_records) != 0):
        status_ddb = get_status_table("us-west-2")
        for id in test_records:
            try:
                status_ddb.delete_status(id)
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
def test_status_entry(id, status, timestamp, teardown):
    status_ddb = get_status_table("us-west-2")
    status_data = create_status_data(id, status, timestamp, teardown)
    output = status_ddb.create_status_entry(status_data)
    assert output == "Success"


@pytest.mark.parametrize(
    "id,status,timestamp",
    [
        (
            None,                                      # missing id
            StatusEnum.SUCCESS.name,
            datetime.datetime.now().isoformat()
        ),
        (
            "2",
            None,                                      # missing status
            datetime.datetime.now().isoformat()
        ),
        (
            "3",
            StatusEnum.SUCCESS.name,
            None                                       # missing timestamp
        ),
        (
            "dummy",                                   # invalid id, should only be numbers in string format (i think)
            StatusEnum.SUCCESS.name,
            datetime.datetime.now().isoformat()
        ),
        (
            5,                                         # invalid id type (should be string)
            StatusEnum.SUCCESS.name,
            datetime.datetime.now().isoformat()
        ),
        (
            "6",
            "random status string",                    # invalid status type (should be one of the elements of StatusEnum)
            datetime.datetime.now().isoformat()
        )
    ]
)
def test_status_entry_invalid_param(id, status, timestamp, teardown):
    status_ddb = get_status_table("us-west-2")
    status_data = create_status_data(id, status, timestamp, teardown)
    with pytest.raises(ValueError) as err:
        status_ddb.create_status_entry(status_data)
    

@pytest.mark.parametrize(
    "id,status,timestamp",
    [
        (
            "1",
            StatusEnum.STARTED.name,
            datetime.datetime.now().isoformat()
        ),
    ]
)    
def test_status_entry_duplicates(id, status, timestamp, teardown):
    status_ddb = get_status_table("us-west-2")
    status_data1 = create_status_data(id, status, timestamp, teardown)
    output1 = status_ddb.create_status_entry(status_data1)
    assert output1 == "Success"
    
    # If request_id already exists, it should not be altered by create_status_entry
    status_data2 = create_status_data(id, status, timestamp, teardown)    
    with pytest.raises(ValueError) as err:
        status_ddb.create_status_entry(status_data2)
    status_ddb.delete_status(id, StatusEnum.SUCCESS.name)


@pytest.mark.parametrize(
    "id",
    [
        "0",
        "1"
    ]
)
def test_get_status(id, teardown):
    status_ddb = get_status_table("us-west-2")
    all_status_data = valid_status_entry_helper(teardown, status_ddb)
    
    output = status_ddb.get_record(id)
    assert output == all_status_data[int(id)]
    
    
@pytest.mark.parametrize(
    "id",
    [
        "4",
        "hello",
        None
    ]
)
def test_get_status_invalid_id(id, teardown):
    status_ddb = get_status_table("us-west-2")
    valid_status_entry_helper(teardown, status_ddb)
    
    with pytest.raises(ValueError) as err:
        status_ddb.get_record(id)
        

@pytest.mark.parametrize(
    "id",
    [
        "4"
    ]
)
def test_get_status_extra_attributes(id, teardown):
    status_ddb = get_status_table("us-west-2")
    data = {"request_id": id, "status": StatusEnum.IN_PROGRESS.name, "timestamp": datetime.datetime.now().isoformat(), "extra": 123}
    teardown.append(id)
    status_ddb.table.put_item(Item=data)
    
    with pytest.raises(ValueError):
        status_ddb.get_record(id)
        
        
@pytest.mark.parametrize(
    "id",
    [
        "1",
        "2"
    ]
)
def test_delete_status(id, teardown):
    status_ddb = get_status_table("us-west-2")
    valid_status_entry_helper(teardown, status_ddb)
    
    output = status_ddb.delete_status(id)
    assert output == "Success"
        
        
#Not sure if it is supposed to accept incorrect ids
@pytest.mark.parametrize(
    "id",
    [
        "4",
        "hello"
    ]
)
def test_delete_status_invalid_id(id, teardown):
    status_ddb = get_status_table("us-west-2")
    valid_status_entry_helper(teardown, status_ddb)
    
    with pytest.raises(ValueError):
        print(status_ddb.delete_status(id))
        

@pytest.mark.parametrize(
    "id,new_status",
    [
        (
            "0",
            StatusEnum.IN_PROGRESS.name
        ),
        (
            "1",
            StatusEnum.FAILED.name
        )
    ]
)
def test_update_status(id, new_status, teardown):
    status_ddb = get_status_table("us-west-2")
    all_status_data = valid_status_entry_helper(teardown, status_ddb)
    
    output = status_ddb.update_status(id, new_status)
    assert output == "Success"
    
    updated_record_from_db = status_ddb.get_record(id)
    original_record = all_status_data[int(id)]
    original_record.status = new_status
    assert updated_record_from_db == original_record
    
    
@pytest.mark.parametrize(
    "id,new_status",
    [
        (
            "dummy",
            StatusEnum.IN_PROGRESS.name
        ),
        (
            None,
            StatusEnum.FAILED.name
        )
    ]
)
def test_update_status_invalid_id(id, new_status, teardown):
    status_ddb = get_status_table("us-west-2")
    valid_status_entry_helper(teardown, status_ddb)
    teardown.append(id)
    with pytest.raises(ValueError):
        status_ddb.update_status(id, new_status)
        

@pytest.mark.parametrize(
    "id,new_status",
    [
        (
            "1",
            "dummy"
        ),
        (
            "2",
            None
        )
    ]
)
def test_update_status_invalid_status(id, new_status, teardown):
    status_ddb = get_status_table("us-west-2")
    valid_status_entry_helper(teardown, status_ddb)
    with pytest.raises(ValueError):
        status_ddb.update_status(id, new_status)