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
    create_env_file(aws_secrets.get_secret_env(aws_secrets.get_secret_response(aws_constants.SECRET_NAME)))


def create_env_file(env_values: Dict[str, str]):
    with open(aws_constants.FINAL_ENV_PATH, "w") as f:
        for key, val in env_values.items():
            f.write(f'{key}="{val}"\n')


def main():
    get_secret()


if __name__ == "__main__":
    main()
