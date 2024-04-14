from django.http import HttpRequest
from ninja.security import HttpBearer
from cli import init_firebase
import firebase_admin.auth
import logging

logger = logging.getLogger()


class FirebaseAuth(HttpBearer):
    def authenticate(self, request, token):
        if token is None or not token:
            return
        app = init_firebase()
        try:
            firebase_admin.auth.verify_id_token(token)
        except Exception as e:
            logger.info(e)
            return
        finally:
            firebase_admin.delete_app(app)
        return token


class Request(HttpRequest):
    auth: str
