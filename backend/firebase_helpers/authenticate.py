import firebase_admin
import firebase_admin.auth
from flask import request
import time

def authenticate(request_data):
    authorization = request_data["authorization"]
    if not authorization:
        return False ## throw some error through middleware
    try:
        user = firebase_admin.auth.verify_id_token(authorization)
        request.user = user
    except Exception as e:
        print(e)
        return False ## throw some error through middleware
    
    return True ## do middleware stuff instead
