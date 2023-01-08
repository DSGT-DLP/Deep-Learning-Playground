from constants import (
    SENDER,
    AWS_REGION,
    CHARSET,
)
import json

import boto3
from botocore.exceptions import ClientError
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import os
import base64

def send_email(email_address,subject="",body_text="",attachment_array=[],fileNames=[]):
    """
    send_email function takes data that comes from API Gateway and uses
    AWS SES to send an email. 

    Args:
        email_address (str): email address of user
        subject (str,optional): subject of the email that needs to be sent
        body_text(str,optional): body of the email that needs to be sent
        attachment_array(array of strings,optional): base64 strings of all the 
        attachmentsfileNames(array of strings, optional): File names of 
        attachments as strings
    """
    
    client = boto3.client("ses", region_name=AWS_REGION)

    msg = MIMEMultipart("mixed")
    msg["Subject"] = subject
    msg["From"] = SENDER
    msg["To"] = email_address

    msg_body = MIMEMultipart("alternative")

    textpart = MIMEText(body_text.encode(CHARSET), "plain", CHARSET)

    msg_body.attach(textpart)

    for i in range(len(attachment_array)):
        data = base64.b64decode(attachment_array[i])
        with open("/tmp/" + fileNames[i], 'wb') as f:
            f.write(data)
        
        att = MIMEApplication(open("/tmp/" + fileNames[i], "rb").read())
        att.add_header(
            "Content-Disposition", 
            "attachment", 
            filename=os.path.basename("/tmp/" + fileNames[i])
        )
        msg.attach(att)
        


    # Attach the multipart/alternative child container to the multipart/mixed
    # parent container.
    msg.attach(msg_body)

    try:
        # Provide the contents of the email.
        response = client.send_raw_email(
            Source=SENDER,
            Destinations=[email_address],
            RawMessage={
                "Data": msg.as_string(),
            },
        )
    # Display an error if something goes wrong.
    except ClientError as e:
        return(e.response["Error"]["Message"])
    else:
        return("Email sent! Message ID:"),
        return(response["MessageId"])

def lambda_handler(event, context):
    print(event)
    myDict = json.loads(event['body'])
    print(event)
    return send_email(event['queryStringParameters']['recipient'],
                      event['queryStringParameters']['subject'],
                      event['queryStringParameters']['body_text'],
                      myDict['attachment_array'],
                      myDict['file_names']
    )
