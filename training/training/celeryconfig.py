broker_url = "sqs://"  # "redis://localhost:6379"

broker_transport_options = {
    "predefined_queues": {
        "training-queue.fifo": {
            "url": "https://sqs.us-east-1.amazonaws.com/521654603461/training-queue.fifo",
        }
    }
}

task_default_queue = "training-queue.fifo"
