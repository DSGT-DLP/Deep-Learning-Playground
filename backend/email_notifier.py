import configparser
import base64
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import (Mail, Attachment, FileContent, FileName, FileType, Disposition)
from constants import ONNX_MODEL, LOSS_VIZ, ACC_VIZ
import requests
import json
import boto3
from botocore.exceptions import ClientError
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import os


def send_email(email):

    """
    If the user inputs a valid email in the frontend, then send_email sends the created ONNX
    file to the user's email using AWS Simple Email Service (SES). 

    Args:
        email (str): email address of user
    """

    # This address must be verified with Amazon SES.
    SENDER = "DSGT Playground <dsgtplayground@gmail.com>"

    # If your account is still in the sandbox, this address must be verified.
    RECIPIENT = email

    AWS_REGION = "us-east-2"

    # The subject line for the email.
    SUBJECT = "Your ONNX file and visualizations from Deep Learning Playground"

    # The full path to the file that will be attached to the email.
    ONNX = ONNX_MODEL
    LOSS = LOSS_VIZ
    ACC = ACC_VIZ

    # The email body for recipients with non-HTML email clients.
    BODY_TEXT = "Attached is the ONNX file and visualizations that you just created in Deep Learning Playground. Please notify us if there are any problems."

    # The HTML body of the email.
    BODY_HTML = """\
    <html>
    <head></head>
    <body>
    <p>Attached is the ONNX file and visualizations that you just created in Deep Learning Playground. Please notify us if there are any problems.</p>
    </body>
    </html>
    """

    CHARSET = "utf-8"

    client = boto3.client('ses',region_name=AWS_REGION)

    msg = MIMEMultipart('mixed')
    msg['Subject'] = SUBJECT 
    msg['From'] = SENDER 
    msg['To'] = RECIPIENT

    msg_body = MIMEMultipart('alternative')

    textpart = MIMEText(BODY_TEXT.encode(CHARSET), 'plain', CHARSET)
    htmlpart = MIMEText(BODY_HTML.encode(CHARSET), 'html', CHARSET)

    msg_body.attach(textpart)
    msg_body.attach(htmlpart)

    onnx_att = MIMEApplication(open(ONNX, 'rb').read())
    onnx_att.add_header('Content-Disposition','attachment',filename=os.path.basename(ONNX))
    loss_att = MIMEApplication(open(LOSS, 'rb').read())
    loss_att.add_header('Content-Disposition','attachment',filename=os.path.basename(LOSS))
    acc_att = MIMEApplication(open(ACC, 'rb').read())
    acc_att.add_header('Content-Disposition','attachment',filename=os.path.basename(ACC))

    # Attach the multipart/alternative child container to the multipart/mixed
    # parent container.
    msg.attach(msg_body)

    # Add the attachment to the parent container.
    msg.attach(onnx_att)
    msg.attach(loss_att)
    msg.attach(acc_att)
    #print(msg)
    try:
        #Provide the contents of the email.
        response = client.send_raw_email(
            Source=SENDER,
            Destinations=[
                RECIPIENT
            ],
            RawMessage={
                'Data':msg.as_string(),
            },
        )
    # Display an error if something goes wrong.	
    except ClientError as e:
        print(e.response['Error']['Message'])
    else:
        print("Email sent! Message ID:"),
        print(response['MessageId'])

