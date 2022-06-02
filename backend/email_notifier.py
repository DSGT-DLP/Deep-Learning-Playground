import configparser
import base64
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import (Mail, Attachment, FileContent, FileName, FileType, Disposition)
from constants import ONNX_MODEL, LOSS_VIZ, ACC_VIZ

def send_email(email):

    """
    If the user inputs a valid email in the frontend, then send_email sends the created ONNX
    file to the user's email using the Sendgrid API. Email may appear in spam folder when 
    sent with Sendgrid.

    Args:
        email (str): email address of user
    """
    attachmentsArr = []
    config = configparser.ConfigParser()
    config.read("config.ini")
    api_key = config["DEFAULT"]["SENDGRID_API_KEY"]
    print(api_key)

    message = Mail(
        from_email="dsgtplayground@gmail.com",
        to_emails=email,
        subject="Your ONNX file and visualizations from Deep Learning Playground",
        html_content="Attached is the ONNX file and visualizations that you just created in Deep Learning Playground. Please notify us if there are any problems.",
    )

    with open(ONNX_MODEL, 'rb') as f:
        data = f.read()
        f.close()

    #ONNX file must be encoded to base64 in order to be sent through email.
    encoded_file = base64.b64encode(data).decode()

    attachedFile = Attachment(
    FileContent(encoded_file),
    FileName('my_deep_learning_model.onnx'),
    FileType('application/onnx'),
    Disposition('attachment')
    )
    attachmentsArr.append(attachedFile)

    with open(LOSS_VIZ, 'rb') as f:
        data = f.read()
        f.close()

    #ONNX file must be encoded to base64 in order to be sent through email.
    encoded_file = base64.b64encode(data).decode()

    attachedFile = Attachment(
    FileContent(encoded_file),
    FileName('loss_visualization.png'),
    FileType('application/png'),
    Disposition('attachment')
    )
    attachmentsArr.append(attachedFile)

    with open(ACC_VIZ, 'rb') as f:
        data = f.read()
        f.close()

    #ONNX file must be encoded to base64 in order to be sent through email.
    encoded_file = base64.b64encode(data).decode()

    attachedFile = Attachment(
    FileContent(encoded_file),
    FileName('accuracy_visualization.png'),
    FileType('application/png'),
    Disposition('attachment')
    )
    attachmentsArr.append(attachedFile)
    message.attachment = attachmentsArr

    try:
        sg = SendGridAPIClient(api_key=api_key)
        response = sg.send(message)
        print(response.status_code)
        print(response.body)
        print(response.headers)
    except Exception as e:
        print(e)
