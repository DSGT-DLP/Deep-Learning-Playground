import boto3
import base64
import json
from botocore.exceptions import ClientError
from typing import Union

from backend.common.constants import AWS_REGION

client = boto3.client('secretsmanager', region_name=AWS_REGION)


def __get_secret_response(secret_name):
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


def get_secret(secret_name: str) -> Union[str, bytes]:
    """
    Returns the secret value from AWS based on the secret name key.

    :param secret_name: Secret name or key
    :return: A string or a byte string representing the secret value; the type
    returned depends on the actual type of the secret value
    """
    secret_response = __get_secret_response(secret_name)
    if 'SecretString' in secret_response:
        return secret_response["SecretString"]
    else:
        return base64.b64decode(secret_response['SecretBinary'])


def create_secret(name, secret, description=''):
    """
    Creates a secret key and value pair in AWS

    :param name: Secret name or key
    :param secret: A dict representing the secret value
    :param description: Optional description of secret
    :return: a dict (JSON) containing the created secret
    """
    return client.create_secret(Name=name, SecretString=json.dumps(secret),
                                Description=description)
