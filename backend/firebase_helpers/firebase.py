import boto3
import json
import firebase_admin

from common.constants import AWS_REGION
from aws_helpers.aws_secrets_utils import aws_secrets


def get_secret():
    return json.loads(aws_secrets.get_secret("DLP/Firebase/Admin_SDK"))


def init_firebase():
    firebase_secret = get_secret()

    # strange bug between aws and python that turned \n into \\n in private_key
    firebase_secret["private_key"] = firebase_secret["private_key"].replace("\\n", "\n")
    firebase_secret["type"] = "service_account"

    creds = firebase_admin.credentials.Certificate(firebase_secret)
    firebase_admin.initialize_app(creds)
