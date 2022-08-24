import boto3
import json
import firebase_admin

from backend.common.constants import AWS_REGION


def get_secret():
    secret_name = "DLP/Firebase/Admin_SDK"

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(service_name="secretsmanager", region_name=AWS_REGION)

    return json.loads(client.get_secret_value(SecretId=secret_name)["SecretString"])

def init_firebase():
  firebase_secret = get_secret()
  firebase_secret["type"] = "service_account"

  creds = firebase_admin.credentials.Certificate(firebase_secret)
  firebase_admin.initialize_app(creds)
