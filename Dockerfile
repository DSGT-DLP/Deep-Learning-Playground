# FROM nikolaik/python3.9-nodejs16
FROM nikolaik/python-nodejs:latest

EXPOSE 8000

WORKDIR /

RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
RUN unzip awscliv2.zip
RUN ./aws/install

ARG AWS_REGION
ARG AWS_DEPLOY_SECRET_ACCESS_KEY
ARG AWS_DEPLOY_ACCESS_KEY_ID

RUN aws configure set region $AWS_REGION
RUN aws configure set aws_access_key_id $AWS_DEPLOY_ACCESS_KEY_ID
RUN aws configure set aws_secret_access_key $AWS_DEPLOY_SECRET_ACCESS_KEY

COPY requirements.txt .

RUN python -m pip install -r requirements.txt

COPY . .

RUN npm run secrets:deploy

RUN npm run build:prod

CMD python -m backend.driver

