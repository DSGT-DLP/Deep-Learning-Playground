FROM python:3.9-slim

WORKDIR /

COPY requirements.txt .
RUN apt-get update -y && apt-get install -y gcc && apt-get install -y curl && apt-get install -y unzip
RUN pip install -r requirements.txt
COPY . .

RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" -o "awscliv2.zip"
RUN unzip awscliv2.zip
RUN ./aws/install

ARG AWS_REGION
ARG AWS_DEPLOY_ACCESS_KEY_ID
ARG AWS_DEPLOY_SECRET_ACCESS_KEY

RUN which aws

RUN aws configure set region $AWS_REGION
RUN aws configure set aws_access_key_id $AWS_DEPLOY_ACCESS_KEY_ID
RUN aws configure set aws_secret_access_key $AWS_DEPLOY_SECRET_ACCESS_KEY

ENV SQS_QUEUE_URL='https://sqs.us-west-2.amazonaws.com/521654603461/dlp-training-queue'

CMD python -m backend.common.kernel