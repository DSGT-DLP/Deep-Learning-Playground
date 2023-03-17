# middleware.py
from werkzeug.wrappers import Request, Response, ResponseStream

from backend.firebase_helpers.authenticate import authenticate
import json


def send_error(environ, start_response):
    response = Response(
        json.dumps({"success": False, "message": "User is not authenticated"}),
        content_type="application/json",
        status=401,
    )
    return response(environ, start_response)


class middleware:
    def __init__(self, app, exempt_paths=[]):
        self.app = app

    def __call__(self, environ, start_response):
        request = Request(environ)
        if (
            "Authorization" in request.headers
            and "bearer " in request.headers["Authorization"]
        ):
            token = request.headers["Authorization"]
            environ["user"] = authenticate(token)
        else:
            return send_error(environ, start_response)

        if not environ["user"]:
            return send_error(environ, start_response)
        return self.app(environ, start_response)