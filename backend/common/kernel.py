import time
import json
import backend.aws_helpers.sqs_utils.sqs_client as sqs_helper

def router(msg):
    '''
    Routes the message to the appropriate training function.
    '''
    if msg['route'] == 'tabular-run':
        # Call the tabular_run() function, location TBD
        _=0
    elif msg['route'] == 'ml-run':
        # Call the ml_run() function, location TBD
        _=0
    elif msg['route'] == 'img-run':
        # Call the img_run() function, location TBD
        _=0
    elif msg['route'] == 'object-detection':
        # Call the object_detection_run() function, location TBD
        _=0
    # if succeed, success status in DDB
    # if fail, fail status in DDB

def empty_message(message):
    '''
    Returns if JSON is empty
    '''
    return json.dumps(message) == "{}"

# Polls for messages from the SQS queue, and handles them.
while True:
    # Get message from queue
    msg = sqs_helper.receive_message()

    if not empty_message(msg):
        print(msg)

        # Update DynamoDB progress
        # - parse message, including execution ID and everything
        # - DynamoDB helper update database function to write to DDB table

        # Handle data
        router(msg)
    else:
        # No message found
        print("No message found")

    # Check again
    time.sleep(1)