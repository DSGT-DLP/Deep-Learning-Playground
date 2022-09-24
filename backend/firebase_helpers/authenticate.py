import firebase_admin
import firebase_admin.auth
from flask import request
import time

from backend.aws_helpers.dynamo_db_utils.user_db import UserDDBUtil, UserData
from backend.common.constants import (
    AWS_REGION,
    USER_TABLE_NAME,
)

userDDBUtil = UserDDBUtil(USER_TABLE_NAME, AWS_REGION)

def authenticate(token):
    user = None
    if token is None or not token:
        return False
    try:
        authorization = token[7:]
        user = firebase_admin.auth.verify_id_token(authorization)

        # create user row in user_db (user-table) if it doesn't exist
        try:
            userDDBUtil.create_record(
                UserData(user["user_id"], user["email"], str(int(time.time())))
            )
        except Exception as e:
            print(e)
            if "Could not add record." not in str(e):
                print(e, "something went wrong in authenticate")
            return user
    except Exception as e:
        print(e)
        return user
    return user
