import firebase_admin
import firebase_admin.auth
from flask import request

from backend.aws_helpers.dynamo_db_utils.user_db import UserDDBUtil, UserData
from backend.common.constants import (
    AWS_REGION,
    USER_TABLE_NAME,
)


userDDBUtil = UserDDBUtil(USER_TABLE_NAME, AWS_REGION)
try:
    userDDBUtil.create_table()  # create table if not exists
except:
    pass


def authenticate(request_data, socket):
    authorization = request_data["authorization"]
    if not authorization:
        socket.emit(
            "authenticationResult", {"success": False, "message": "No token provided"}
        )
        return False
    try:
        user = firebase_admin.auth.verify_id_token(authorization)
        request.user = user

        # create user row in user_db (user-table) if it doesn't exist
        try:
            userDDBUtil.create_record(
                UserData(user["user_id"], user["email"], "", "", "", "")
            )
        except Exception as e:
            if "Could not add record." not in str(e):
                print(e, "something went wrong in authenticate")
                socket.emit(
                    "authenticationResult",
                    {"success": False, "message": "Something went wrong while authenticating"},
                )
            return False
    except Exception as e:
        print(e)
        socket.emit(
            "authenticationResult",
            {"success": False, "message": "Invalid token provided"},
        )
        return False
    return True
