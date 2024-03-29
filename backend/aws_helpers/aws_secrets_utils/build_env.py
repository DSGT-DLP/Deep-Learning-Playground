# Use this code snippet in your app.
# If you need more information about configurations or implementing the sample code, visit the AWS docs:
# https://aws.amazon.com/developers/getting-started/python/

import boto3
import base64
import os
import json
from botocore.exceptions import ClientError
from typing import Dict
import sys

from aws_helpers.aws_secrets_utils import aws_secrets
from aws_helpers.aws_secrets_utils import aws_constants


def get_secret():
    create_env_file(json.loads(aws_secrets.get_secret(aws_constants.SECRET_NAME)))


def create_env_file(env_values: Dict[str, str]):
    with open(aws_constants.FINAL_ENV_PATH, "w") as f:
        for key, val in env_values.items():
            f.write(f'{key}="{val}"\n')


def main():
    get_secret()


if __name__ == "__main__":
    main()
