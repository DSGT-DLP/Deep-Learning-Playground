from constants import ONNX_MODEL, LOSS_VIZ, ACC_VIZ, SENDER, AWS_REGION, BODY_TEXT, BODY_HTML, CHARSET
import boto3
from botocore.exceptions import ClientError
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import os


def send_email(email):
    """
    If the user inputs a valid email in the frontend, then send_email sends the created ONNX
    file to the user's email using AWS Simple Email Service (SES). Use AWS CLI to configure 
    AWS key and secret key in order for this function to run.

    Args:
        email (str): email address of user
    """

    client = boto3.client('ses',region_name=AWS_REGION)

    msg = MIMEMultipart('mixed')
    msg['Subject'] = "Your ONNX file and visualizations from Deep Learning Playground"
    msg['From'] = SENDER 
    msg['To'] = email

    msg_body = MIMEMultipart('alternative')

    textpart = MIMEText(BODY_TEXT.encode(CHARSET), 'plain', CHARSET)
    htmlpart = MIMEText(BODY_HTML.encode(CHARSET), 'html', CHARSET)

    msg_body.attach(textpart)
    msg_body.attach(htmlpart)

    onnx_att = MIMEApplication(open(ONNX_MODEL, 'rb').read())
    onnx_att.add_header('Content-Disposition','attachment',filename=os.path.basename(ONNX_MODEL))
    loss_att = MIMEApplication(open(LOSS_VIZ, 'rb').read())
    loss_att.add_header('Content-Disposition','attachment',filename=os.path.basename(LOSS_VIZ))
    acc_att = MIMEApplication(open(ACC_VIZ, 'rb').read())
    acc_att.add_header('Content-Disposition','attachment',filename=os.path.basename(ACC_VIZ))

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
                email
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

