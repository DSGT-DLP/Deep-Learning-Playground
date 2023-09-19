from flask import Blueprint

test_bp = Blueprint("test", __name__)


@test_bp.route("/", methods=["GET"])
def verify_backend_alive():
    """
    Dummy API endpoint to verify that you can make a simple backend API request

    Params:
     - None

    Results:
     - 200: you can see the JSON {"Status": "Backend is alive"}
     - 400: something went wrong in getting the result
     - 404: Not authorized to access this endpoint
    """
    return {"Status": "Backend is alive"}
