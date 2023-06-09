import traceback

from flask import Blueprint
from flask import request
import json

from backendCore.endpoints.utils import send_success, send_traceback_error

sqs_bp = Blueprint("sqs", __name__)


@sqs_bp.route("/queue", methods=["POST"])
def add_to_queue():
    request_data = json.loads(request.data)
    queue_name = request_data["queue_name"]
    body = request_data["body"]

    print("Adding to queue: ", queue_name, body)

    try:
        return send_success({"message": "Sent email to "})
    except Exception:
        print(traceback.format_exc())
        return send_traceback_error()
