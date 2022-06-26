import pytest
from backend import email_notifier as em
import json

@pytest.mark.parametrize(
    "email,subject,body_text",
    [
        (
            "",
            "hello",
            "body"
        ),
        (
            "dsgtplayground@gmail",
            "subject",
            "body"
        ),
    ],
)
def test_send_email_fail(email, subject,body_text):
    #{"message":"Internal Server Error"}
    #["Email sent! Message ID:"]
    with pytest.raises(ValueError):
        response = em.send_email(email,subject,body_text)
    #print(response)
    #print(response.text)
    #assert json.loads(response.text)["message"] == "Internal Server Error"


@pytest.mark.parametrize(
    "email,subject,body_text",
    [
        (
            "karkir0003@outlook.com",
            "subject",
            "body"
        )
    ],
)
def test_send_email_success(email,subject,body_text):
    response = em.send_email(email,subject,body_text)
    assert json.loads(response.text)[0] == "Email sent! Message ID:"

@pytest.mark.parametrize(
    "email,subject,body_text,attachment_array",
    [
        (
            "karkir0003@outlook.com",
            "subject",
            "body",
            ["frontend/playground-frontend/src/images/logos/react-logo.png"]
        )
    ],
)
def test_send_email_attachment(email,subject,body_text,attachment_array):
    response = em.send_email(email,subject,body_text,attachment_array)
    assert json.loads(response.text)[0] == "Email sent! Message ID:"
