import configparser
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

config = configparser.ConfigParser()
config.read("config.ini")
api_key = config["DEFAULT"]["SENDGRID_API_KEY"]
print(f"api key: {api_key}")
message = Mail(
    from_email="spendtrack671@egmail.com",
    to_emails="karkir0003@gmail.com",
    subject="Sending with Twilio SendGrid is Fun",
    html_content="<strong>and easy to do anywhere, even with Python</strong>",
)
try:
    sg = SendGridAPIClient(api_key=api_key)
    response = sg.send(message)
    print(response.status_code)
    print(response.body)
    print(response.headers)
except Exception as e:
    print(e)
