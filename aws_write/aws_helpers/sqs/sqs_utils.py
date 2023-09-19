import boto3
from common.constants import AWS_REGION
import json
from aws_helpers.sqs.constants import TRAINING_QUEUE

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
