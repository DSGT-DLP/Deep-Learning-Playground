import traceback

from flask import Blueprint
from flask import request

from aws_helpers.lambda_utils.lambda_client import invoke
from aws_helpers.sqs_utils.sqs_client import add_to_training_queue
from common.email_notifier import send_email
from common.utils import *
from endpoints.utils import send_error, send_success, send_traceback_error

aws_bp = Blueprint("aws", __name__)


@aws_bp.route("/sendEmail", methods=["POST"])
def send_email_route():
    """
    API Endpoint to send email notification via AWS SES.

    This endpoint applies for users submitting the feedback form

    Request Data:
      - email_address: email address of the sender
      - subject: Subject line of email
      - body_text: Content of the email

    Results:
      - 200: Feedback form submitted successfully
      - 400: Error in submitting feedback form
    """
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
    """
    API Endpoint that sends a user's custom data preprocessing code to an AWS lambda endpoint
    that executes it

    Params:
     - data: Dataset the user wants to preprocess
     - codeSnippet: preprocessing code

    Results:
     - 200: Preprocessing done successfully
     - 400: Something went wrong in preprocessing the data with the user supplied code
    """
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
