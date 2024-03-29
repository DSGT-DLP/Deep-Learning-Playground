import firebase_admin
import firebase_admin.auth


def authenticate(token):
    user = None
    if token is None or not token:
        return False
    try:
        authorization = token[7:]
        user = firebase_admin.auth.verify_id_token(authorization)
    except Exception as e:
        print(e)
        return None

    return user
