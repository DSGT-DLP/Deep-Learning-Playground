from django.http import HttpRequest
from ninja.security import HttpBearer
from cli import init_firebase
import firebase_admin.auth
import logging

logger = logging.getLogger()

class FirebaseAuth(HttpBearer):
    def authenticate(self, request, token):
        app = init_firebase()
        if token is None or not token:
            return
        try:
            firebase_admin.auth.verify_id_token(token)
            firebase_admin.delete_app(app)
        except Exception as e:
            logger.info(e)
            return
        return token


class Request(HttpRequest):
    auth: str
