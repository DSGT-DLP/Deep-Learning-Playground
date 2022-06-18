from constants import (
    SENDER,
    AWS_REGION,
    CHARSET,
)
import boto3
from botocore.exceptions import ClientError
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import os


def send_email(email_address,subject,body_text,attachment_array):
    """
    If the user inputs a valid email in the frontend, then send_email sends the created ONNX
    file to the user's email using AWS Simple Email Service (SES). Use AWS CLI to configure
    AWS key and secret key in order for this function to run.

    Args:
        email_address (str): email address of user
        subject (str): subject of the email that needs to be sent
        body_text(str): body of the email that needs to be sent
        attachement_array(array of strings): filepaths of the attachements that need to be sent
    """

    client = boto3.client("ses", region_name=AWS_REGION)

    msg = MIMEMultipart("mixed")
    msg["Subject"] = subject
    msg["From"] = SENDER
    msg["To"] = email_address

    msg_body = MIMEMultipart("alternative")

    textpart = MIMEText(body_text.encode(CHARSET), "plain", CHARSET)

    msg_body.attach(textpart)

    for attachment in attachment_array:
        att = MIMEApplication(open(attachment, "rb").read())
        att.add_header(
            "Content-Disposition", "attachment", filename=os.path.basename(attachment)
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
        print(e.response["Error"]["Message"])
    else:
        print("Email sent! Message ID:"),
        print(response["MessageId"])
