import pytest
from backend.common import email_notifier as em
import json

SUCCESS_EMAIL = "Email sent! Message ID:"


@pytest.mark.parametrize(
    "email,subject,body_text",
    [
        ("", "hello", "body"),
        ("dsgtplayground@gmail", "subject", "body"),
    ],
)
def test_send_email_fail(email, subject, body_text):
    with pytest.raises(ValueError):
        response = em.send_email(email, subject, body_text)


@pytest.mark.parametrize(
    "email,subject,body_text",
    [("karkir0003@outlook.com", "subject", "body")],
)
def test_send_email_success(email, subject, body_text):
    response = em.send_email(email, subject, body_text)
    assert response.text == SUCCESS_EMAIL


@pytest.mark.parametrize(
    "email,subject,body_text,attachment_array",
    [
        (
            "karkir0003@outlook.com",
            "subject",
            "body",
            ["frontend/layer_docs/softmax_equation.png"],
        )
    ],
)
def test_send_email_attachment(email, subject, body_text, attachment_array):
    response = em.send_email(email, subject, body_text, attachment_array)
    assert response.text == SUCCESS_EMAIL
