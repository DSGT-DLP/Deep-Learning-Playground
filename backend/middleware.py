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


# to help render the apidocs endpoint with flasgger
SWAGGER_UI_ROUTES = [
    "/flasgger_static/swagger-ui.css",
    "/flasgger_static/swagger-ui-bundle.js",
    "/flasgger_static/swagger-ui-standalone-preset.js",
    "/flasgger_static/lib/jquery.min.js",
    "/flasgger_static/favicon-32x32.png",
]


class middleware:
    def __init__(self, app, exempt_paths=[]):
        self.app = app
        self.exempt_paths = exempt_paths

    def __call__(self, environ, start_response):
        request = Request(environ)
        # '/api/defaultDataset' => '/defaultDataset'
        real_route = "/" + "".join(request.path.split("/")[2:])
        if request.path in SWAGGER_UI_ROUTES or real_route in self.exempt_paths:
            print("in exempt paths")
            return self.app(environ, start_response)
        if (
            "Authorization" in request.headers
            and "Bearer " in request.headers["Authorization"]
        ):
            token = request.headers["Authorization"]
            environ["user"] = authenticate(token)
        else:
            return send_error(environ, start_response)

        if not environ["user"]:
            return send_error(environ, start_response)
        return self.app(environ, start_response)
