"""
File that houses helper functions for testing the training backend 
"""
import jwt
import datetime


def mock_authenticate(*args, **kwargs) -> str:
    """
    Function that gives a test JWT Token for testing (not necessarily real user data)
    Django API Endpoints that require user authentication

    Returns:
        token: Bearer Token
    """
    payload = {
        "sub": "1234567890",
        "name": "John Doe",
        "iat": datetime.datetime.utcnow(),
        "exp": datetime.datetime.utcnow() + datetime.timedelta(days=1),
    }
    secret = "secret"
    token = jwt.encode(payload, secret, algorithm="HS256")
    return token


def get_test_bearer_token() -> dict:
    """
    Wrapper that uses mock_authenticate function to build a bearer token
    in a format that Django accepts
    """
    return {"HTTP_AUTHORIZATION": "Bearer " + mock_authenticate()}
