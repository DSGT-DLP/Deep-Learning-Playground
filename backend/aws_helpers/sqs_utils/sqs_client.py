import boto3
import json
from botocore.exceptions import ClientError
from backend.aws_helpers.sqs_utils.constants import TRAINING_QUEUE
from backend.common.constants import AWS_REGION

"""
Wrapper to interface with AWS SQS (Simple Queue Service)

Helpful Resource: https://www.learnaws.org/2020/12/17/aws-sqs-boto3-guide/
"""

sqs_client = boto3.client("sqs", region_name=AWS_REGION)


def get_queue_url(queue_name=TRAINING_QUEUE):
    """
    Given name of SQS queue, get the queue url

    Args:
        queue_name (str, optional): Name of SQS queue. Defaults to TRAINING_QUEUE.
    """
    response = sqs_client.get_queue_url(QueueName=queue_name)
    return response["QueueUrl"]


def add_to_queue(queue_name, body):
    """
    Add entry to the queue

    Args:
        queue_name (str): name of SQS queue
        body (json): entry to be added to sqs queue
    """
    queue_url = get_queue_url(queue_name)
    response = sqs_client.send_message(QueueUrl=queue_url, MessageBody=json.dumps(body))
    return response


def add_to_training_queue(body):
    """
    Convenient function to add training request parameters into training queue

    Args:
        body (json): training request parameters
    """
    return add_to_queue(TRAINING_QUEUE, body)


def delete_message(queue_name, receipt_handle):
    """
    Delete message from queue given receipt handle

    Args:
        queue_name (str): name of SQS queue
        receipt_handle (json): receipt handle that comes from calling sqs_client.receive_message()
    """
    queue_url = get_queue_url(queue_name)
    response = sqs_client.delete_message(
        QueueUrl=queue_url, ReceiptHandle=receipt_handle
    )
    status_code = response["ResponseMetadata"]["HTTPStatusCode"]
    return json.loads(json.dumps({"status_code": status_code}))


def receive_message(queue_name=TRAINING_QUEUE):
    """
    Utility function to receive message from SQS queue in order to process

    Args:
        queue_name (str): name of SQS queue

    """

    queue_url = get_queue_url(queue_name)
    response = sqs_client.receive_message(
        QueueUrl=queue_url, MaxNumberOfMessages=1, WaitTimeSeconds=10
    )

    messages = response.get("Messages", [])
    if len(messages) == 0:
        return json.loads("{}")  # no messages received

    message_body = json.loads(messages[0]["Body"])
    receipt_handle = messages[0]["ReceiptHandle"]

    # delete received message for safety purposes
    delete_result = delete_message(queue_name, receipt_handle)
    if delete_result["status_code"] == 200:
        return message_body
    return json.loads("{}")


def receive_training_queue_message():
    """
    Convenient utility function to receive message from training queue
    """
    return receive_message(TRAINING_QUEUE)
