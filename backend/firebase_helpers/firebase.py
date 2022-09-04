import boto3
import json
import firebase_admin

from backend.common.constants import AWS_REGION
from backend.aws_helpers.aws_secrets_utils import aws_secrets

def get_secret():
    return aws_secrets.get_secret_string_json(aws_secrets.get_secret_response("DLP/Firebase/Admin_SDK"))

def init_firebase():
  firebase_secret = get_secret()
  firebase_secret["type"] = "service_account"

  creds = firebase_admin.credentials.Certificate(firebase_secret)
  firebase_admin.initialize_app(creds)
