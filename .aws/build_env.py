# Use this code snippet in your app.
# If you need more information about configurations or implementing the sample code, visit the AWS docs:
# https://aws.amazon.com/developers/getting-started/python/

import boto3
import base64
import os
from botocore.exceptions import ClientError
from typing import Dict

ENV_KEYS = ["REACT_APP_SECRET_KEY",
            "REACT_APP_CAPTCHA_SITE_KEY", "REACT_APP_FEEDBACK_EMAIL"]
FINAL_ENV_PATH = os.path.abspath(os.path.join(
    os.getcwd(), '..', 'frontend', 'playground-frontend', '.env'))


def get_secret():
    secret_name = "frontend_env"
    region_name = "us-west-2"

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    # In this sample we only handle the specific exceptions for the 'GetSecretValue' API.
    # See https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
    # We rethrow the exception by default.

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
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
    else:
        # Decrypts secret using the associated KMS key.
        # Depending on whether the secret is a string or binary, one of these fields will be populated.
        env_values = {}
        if 'SecretString' in get_secret_value_response:
            env_values = eval(get_secret_value_response['SecretString'])
        else:
            env_values = eval(base64.b64decode(
                get_secret_value_response['SecretBinary']))

        create_env_file(env_values)


def create_env_file(env_values: Dict[str, str]):
    with open(FINAL_ENV_PATH, "w") as f:
        for key, val in env_values.items():
            f.write(f"{key}={val}\n")


def main():
    get_secret()


if __name__ == "__main__":
    main()
