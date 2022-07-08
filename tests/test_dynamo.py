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

@pytest.mark.parametrize(
    "id,status,timestamp",
    [
        (
            "dummy",
            StatusEnum.STARTED.name,
            datetime.datetime.now()
        ),
        (
            "dummy",
            StatusEnum.SUCCESS.name,
            datetime.datetime.now()
        ),
    ],
)
def test_status_entry(id,status,timestamp):
    status_ddb = get_status_table("us-west-2")
    dummy_status = {"request_id": "dummy", "status": StatusEnum.STARTED.name, "timestamp": datetime.datetime.now().isoformat()}
    status_data = StatusData(set_status_data(dummy_status, StatusAttribute.REQUEST_ID),
                                set_status_data(dummy_status, StatusAttribute.STATUS),
                                set_status_data(dummy_status, StatusAttribute.TIMESTAMP))
    output = status_ddb.create_status_entry(status_data)
    status_ddb.delete_status("dummy",StatusEnum.SUCCESS.name)

    assert output == "Success"
    