# FROM nikolaik/python3.9-nodejs16
FROM nikolaik/python-nodejs:latest

EXPOSE 8000

WORKDIR /

RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
RUN unzip awscliv2.zip
RUN sudo ./aws/install

ARG AWS_REGION=""
ARG AWS_DEPLOY_SECRET_ACCESS_KEY=""
ARG AWS_DEPLOY_ACCESS_KEY_ID=""

RUN aws configure set region $AWS_REGION aws_access_key_id $AWS_DEPLOY_ACCESS_KEY_ID aws_secret_access_key $AWS_DEPLOY_SECRET_ACCESS_KEY

COPY requirements.txt .

RUN python -m pip install -r requirements.txt

COPY . .

RUN cd .aws && python build_env.py

RUN cd frontend/playground-frontend && npm install && npm run build

CMD python -m backend.driver

