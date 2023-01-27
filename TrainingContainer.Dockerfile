FROM python:3.9-slim

COPY requirements.txt .
RUN apt-get update -y && apt-get install -y gcc
RUN pip install -r requirements.txt
COPY . .

ENV SQS_QUEUE_URL='https://sqs.us-west-2.amazonaws.com/521654603461/dlp-training-queue'

CMD python -m backend.poller