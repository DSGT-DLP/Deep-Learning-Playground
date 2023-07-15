import boto3
import json
from backend.common.constants import AWS_REGION

client = boto3.client("lambda", region_name=AWS_REGION)


def invoke(function_name, payload, invocation_type="RequestResponse", log_type="Tail"):
    response = client.invoke(
        FunctionName=function_name,
        InvocationType=invocation_type,
        LogType=log_type,
        Payload=payload,
    )
    return json.loads(response["Payload"].read())
