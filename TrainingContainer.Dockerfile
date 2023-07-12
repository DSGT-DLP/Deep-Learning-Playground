FROM python:3.9-slim

WORKDIR /

# Install Poetry and project dependencies
RUN curl -sSL https://install.python-poetry.org | python -
COPY pyproject.toml ./
RUN poetry install --no-interaction --no-ansi --no-root

COPY . .
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

CMD poetry run python -m backend.common.kernel