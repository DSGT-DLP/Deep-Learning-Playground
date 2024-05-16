import os

# note, this file is used both in training/ and in the celery worker
if "ENVIRONMENT" in os.environ and os.environ["ENVIRONMENT"] == "production":
    print(
        "ENVIRONMENT env var set to production, setting broker_url to sqs training-queue"
    )
    broker_url = "sqs://"

    broker_transport_options = {
        "predefined_queues": {
            "training-queue.fifo": {
                "url": "https://sqs.us-east-1.amazonaws.com/521654603461/training-queue.fifo",
            }
        }
    }

    task_default_queue = "training-queue.fifo"
else:
    print(
        "ENVIRONMENT env var either not found or not set to production, setting broker_url to redis://redis:6379"
    )
    broker_url = "redis://redis:6379"
