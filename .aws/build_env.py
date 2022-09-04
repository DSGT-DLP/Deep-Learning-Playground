# Use this code snippet in your app.
# If you need more information about configurations or implementing the sample code, visit the AWS docs:
# https://aws.amazon.com/developers/getting-started/python/

import boto3
import base64
import os
from botocore.exceptions import ClientError
from typing import Dict
from backend.aws_helpers.aws_secrets_utils import aws_secrets
import aws_constants

def get_secret():
    try:
        secret_response = aws_secrets.get_secret_response(aws_constants.SECRET_NAME)
    except Exception: raise
    else:
        # Decrypts secret using the associated KMS key.
        # Depending on whether the secret is a string or binary, one of these fields will be populated.
        env_values = {}
        if aws_secrets.has_secret_string(secret_response):
            env_values = eval(aws_secrets.get_secret_string(secret_response))
        else:
            env_values = eval(aws_secrets.get_secret_binary_decoded(secret_response))

        create_env_file(env_values)


def create_env_file(env_values: Dict[str, str]):
    with open(aws_constants.FINAL_ENV_PATH, "w") as f:
        for key, val in env_values.items():
            f.write(f'{key}="{val}"\n')


def main():
    get_secret()


if __name__ == "__main__":
    main()
