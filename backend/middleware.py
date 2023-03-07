from werkzeug.wrappers import Request, Response, ResponseStream
from werkzeug.routing import Map, Rule
from backend.firebase_helpers.authenticate import authenticate
import json

#Make sure that this is unique!
url_map = Map([
    Rule('/test', endpoint='test'),
])

def send_error(environ, start_response):
    response = Response(
        json.dumps({"success": False, "message": "User is not authenticated"}),
        content_type="application/json",
        status=401,
    )
    return response(environ, start_response)


class middleware:
    def __init__(self, app, exempt_paths=None):
        self.app = app
        self.exempt_paths=exempt_paths
        self.url_map = self.build_url_map(exempt_paths)
        
    def build_url_map(self, exempt_paths):
        url_map = Map([])
        for path in exempt_paths:
            url_map.add(Rule(path, endpoint=path[1:])) #strip off the "/"
        return url_map
    
    def __call__(self, environ, start_response):
        try:
            request = Request(environ)
            rule = self.url_map.bind_to_environ(environ).match()
            endpoint = rule[0]
            print(f"endpoint: {endpoint}")
            if (endpoint is not None):
                return self.app(environ, start_response) #exempt paths are publicly accessible
        except Exception as e:
            return send_error(environ, start_response)
        
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
