FROM nikolaik/python-nodejs:python3.9-nodejs18

EXPOSE 8000

WORKDIR /
ARG TARGETARCH

RUN if [ "${TARGETARCH}" = "arm64" ] ; then \
        curl "https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" -o "awscliv2.zip" ; \
    else \
        curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" ; \
    fi

RUN unzip awscliv2.zip
RUN ./aws/install

ARG AWS_REGION
ARG AWS_DEPLOY_SECRET_ACCESS_KEY
ARG AWS_DEPLOY_ACCESS_KEY_ID

RUN aws configure set region $AWS_REGION
RUN aws configure set aws_access_key_id $AWS_DEPLOY_ACCESS_KEY_ID
RUN aws configure set aws_secret_access_key $AWS_DEPLOY_SECRET_ACCESS_KEY

# Install Poetry and project (prod) dependencies
RUN curl -sSL https://install.python-poetry.org | python -
COPY backend/pyproject.toml backend/poetry.lock ./backend/
WORKDIR /backend
RUN poetry install --no-interaction --no-ansi --no-root --no-dev

# Switch back to root directory
WORKDIR /
COPY . .

RUN yarn run secrets
RUN yarn run build:prod
CMD yarn run startb