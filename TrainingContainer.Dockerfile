FROM python:3.9-slim

WORKDIR /
# Install Poetry
RUN pip install poetry

# Install gcc
RUN apt-get update && apt-get install -y gcc

# Install curl
RUN apt-get install -y curl

# Install unzip
RUN apt-get install -y unzip

# Checking poetry version
RUN poetry --version

# Copy the project files
COPY backend/pyproject.toml backend/poetry.lock ./backend/

# Change working directory to the backend subdirectory
WORKDIR /backend

# Install prod dependencies
RUN poetry install --no-interaction --no-ansi --no-root --no-dev

# Copy the rest of the project
COPY backend/ ./

ARG TARGETARCH

RUN if [ "${TARGETARCH}" = "arm64" ] ; then \
        curl "https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" -o "awscliv2.zip" ; \
    else \
        curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" ; \
    fi

RUN unzip awscliv2.zip
RUN ./aws/install

ARG AWS_REGION
ARG AWS_DEPLOY_ACCESS_KEY_ID
ARG AWS_DEPLOY_SECRET_ACCESS_KEY

RUN aws configure set region $AWS_REGION
RUN aws configure set aws_access_key_id $AWS_DEPLOY_ACCESS_KEY_ID
RUN aws configure set aws_secret_access_key $AWS_DEPLOY_SECRET_ACCESS_KEY

ENV SQS_QUEUE_URL='https://sqs.us-west-2.amazonaws.com/521654603461/dlp-training-queue'

# Set the working directory to /backend
WORKDIR /backend

CMD poetry run python -m common.kernel