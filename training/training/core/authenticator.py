from ninja.security import HttpBearer
import firebase_admin


class FirebaseAuth(HttpBearer):
    def authenticate(self, request, token):
        if token is None or not token:
            return
        try:
            firebase_admin.auth.verify_id_token(token)
        except Exception as e:
            print(e)
            return
        return token


class Request(HttpRequest):
    auth: str
