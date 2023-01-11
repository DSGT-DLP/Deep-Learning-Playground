import base64
import requests
import re


def send_email(email_address, subject="", body_text="", attachment_array=[]):
    """
    send_email function takes data about  email and sends a post request
    to API gateway which calls AWS Lambda which calls AWS SES to then
    send an email to email_address.
    Args:
        email_address (str): email address of user
        subject (str,optional): subject of the email that needs to be sent
        body_text(str,optional): body of the email that needs to be sent
        attachment_array(array of strings,optional): file paths as strings
    """
    regex = re.compile(
        r"([A-Za-z0-9]+[.-_])*[A-Za-z0-9]+@[A-Za-z0-9-]+(\.[A-Z|a-z]{2,})+"
    )
    print("I'm before full match")
    if not re.fullmatch(regex, email_address):
        raise ValueError("Please enter a valid email to the send_email function")
    fileNames = [fileName.split("/")[-1] for fileName in attachment_array]
    base64Array = []
    for attachment in attachment_array:
        with open(attachment, "rb") as file:
            my_string = base64.b64encode(file.read())
            my_string = my_string.decode("utf-8")
        base64Array.append(my_string)

    url = "https://6amfyprxh9.execute-api.us-west-2.amazonaws.com/default/send_email"
    params = {"recipient": email_address, "subject": subject, "body_text": body_text}
    body = {"attachment_array": base64Array, "file_names": fileNames}
    print("Before post")
    post = requests.post(url, params=params, json=body)
    return post
