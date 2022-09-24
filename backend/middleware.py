from werkzeug.wrappers import Request, Response, ResponseStream

from backend.firebase_helpers.authenticate import authenticate

class middleware():
  def __init__(self, app):
    self.app = app
  
  def __call__(self, environ, start_response):
    request = Request(environ)
    if 'Authorization' in request.headers and 'bearer ' in request.headers['Authorization']:
      token = request.headers['Authorization']
      environ['user'] = authenticate(token)
    return self.app(environ, start_response)
