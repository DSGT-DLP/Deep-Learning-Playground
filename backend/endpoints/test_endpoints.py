from flask import Blueprint

test_bp = Blueprint("test", __name__)


@test_bp.route("/", methods=["GET"])
def verify_backend_alive():
    return {"Status": "Backend is alive"}
