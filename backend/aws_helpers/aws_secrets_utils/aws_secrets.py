import boto3
import json

from backend.common.constants import AWS_REGION

client = boto3.client('secretsmanager', region_name=AWS_REGION)

def get_secret(secret_name):
    return json.loads(get_secret_res(secret_name)["SecretString"])

def get_secret_res(secret_name):
    return client.get_secret_value(SecretId=secret_name)

def create_secret(name, secret, description=''):
    return client.create_secret(Name=name, SecretString=json.dumps(secret), Description=description)

def delete_secret(secret_id):
    return client.delete_secret(SecretId=secret_id)

create_secret("DUMMY_SECRET", {"username":"david","password":"EXAMPLE-PASSWORD"}, "dummy secret purely for testing purposes")

print(get_secret())