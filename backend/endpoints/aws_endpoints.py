import traceback

from flask import Blueprint
from flask import request

from backend.aws_helpers.lambda_utils.lambda_client import invoke
from backend.aws_helpers.sqs_utils.sqs_client import add_to_training_queue
from backend.common.email_notifier import send_email
from backend.common.utils import *
from backend.endpoints.utils import (
    send_error,
    send_success,
    send_traceback_error,
    createExecution,
)

aws_bp = Blueprint("aws", __name__)


@aws_bp.route("/sendEmail", methods=["POST"])
def send_email_route():
    # extract data
    request_data = json.loads(request.data)
    required_params = ["email_address", "subject", "body_text"]
    for required_param in required_params:
        if required_param not in request_data:
            return send_error("Missing parameter " + required_param)

    email_address = request_data["email_address"]
    subject = request_data["subject"]
    body_text = request_data["body_text"]
    if "attachment_array" in request_data:
        attachment_array = request_data["attachment_array"]
        if not isinstance(attachment_array, list):
            return send_error("Attachment array must be a list of filepaths")
    else:
        attachment_array = []

    # try to send email
    try:
        send_email(email_address, subject, body_text, attachment_array)
        return send_success({"message": "Sent email to " + email_address})
    except Exception:
        print(traceback.format_exc())
        return send_traceback_error()


@aws_bp.route("/sendUserCodeEval", methods=["POST"])
def send_user_code_eval():
    try:
        request_data = json.loads(request.data)
        data = request_data["data"]
        codeSnippet = request_data["codeSnippet"]
        payload = json.dumps({"data": data, "code": codeSnippet})
        resJson = invoke("preprocess_data", payload)
        if resJson["statusCode"] == 200:
            send_success(
                {
                    "message": "Preprocessed data",
                    "data": resJson["data"],
                    "columns": resJson["columns"],
                }
            )
        else:
            print(resJson["message"])
            send_error(resJson["message"])
        return resJson
    except Exception:
        print(traceback.format_exc())
        print("error")
        print("Last element: ", send_traceback_error()[0])
        return send_traceback_error()


@aws_bp.route("/writeToQueue", methods=["POST"])
def writeToQueue() -> str:
    """
    API Endpoint to write training request to SQS queue to be serviced by
    ECS Fargate training cluster

    """
    try:
        queue_data = json.loads(request.data)
        queue_send_outcome = add_to_training_queue(queue_data)
        print(f"sqs outcome: {queue_send_outcome}")
        status_code = queue_send_outcome["ResponseMetadata"]["HTTPStatusCode"]
        if status_code != 200:
            return send_error("Your training request couldn't be added to the queue")
        else:
            createExecution(queue_data)
            return send_success(
                {"message": "Successfully added your training request to the queue"}
            )
    except Exception:
        return send_error("Failed to queue data")
