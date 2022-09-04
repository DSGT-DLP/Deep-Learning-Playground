import boto3
import base64
import json
from botocore.exceptions import ClientError

from backend.common.constants import AWS_REGION

client = boto3.client('secretsmanager', region_name=AWS_REGION)

def get_secret_response(secret_name):
    try:
        return client.get_secret_value(SecretId=secret_name)
    # In this sample we only handle the specific exceptions for the 'GetSecretValue' API.
    # See https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
    # We rethrow the exception by default.
    except ClientError as e:
        if e.response['Error']['Code'] == 'DecryptionFailureException':
            # Secrets Manager can't decrypt the protected secret text using the provided KMS key.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response['Error']['Code'] == 'InternalServiceErrorException':
            # An error occurred on the server side.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response['Error']['Code'] == 'InvalidParameterException':
            # You provided an invalid value for a parameter.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response['Error']['Code'] == 'InvalidRequestException':
            # You provided a parameter value that is not valid for the current state of the resource.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response['Error']['Code'] == 'ResourceNotFoundException':
            # We can't find the resource that you asked for.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e

def get_secret_string(secret_response):
    return secret_response["SecretString"]

def get_secret_string_json(secret_response):
    return json.loads(get_secret_string(secret_response))

def has_secret_string(secret_response):
    return 'SecretString' in secret_response

def get_secret_binary_decoded(secret_response):
    return base64.b64decode(secret_response['SecretBinary'])

def create_secret(name, secret, description=''):
    return client.create_secret(Name=name, SecretString=json.dumps(secret), Description=description)

def delete_secret(secret_id):
    return client.delete_secret(SecretId=secret_id)