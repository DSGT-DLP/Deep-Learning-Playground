import json
from firebase_admin import auth
import firebase_admin
import boto3
from botocore.exceptions import ClientError
import click
import requests

def init_firebase():
    secret_name = "DLP/Firebase/Admin_SDK"
    region_name = "us-east-1"
    # Create a Secrets Manager client

    client = boto3.client("secretsmanager", region_name=region_name)
    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
    # For a list of exceptions thrown, see
    # https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
    except ClientError as e:
        if 'UnrecognizedClientException' in str(e.response['Error']['Code']):
            raise RuntimeError("AWS authentification incomplete. Make sure all credentials are set correctly including `export AWS_PROFILE=<profile-name>`")
        raise e

    # Decrypts secret using the associated KMS key.
    secret_str = get_secret_value_response["SecretString"]
    secret = json.loads(secret_str)
    secret["private_key"] = secret["private_key"].replace("\\n", "\n")
    secret["type"] = "service_account"
    credential = firebase_admin.credentials.Certificate(secret)
    return firebase_admin.initialize_app(credential)


@click.group()
def cli():
    pass


@cli.command()
@click.argument("email")
def get_uid(email: str) -> None:
    """Gets a user's uid given their email."""

    app = init_firebase()
    click.echo(auth.get_user_by_email(email).uid)
    firebase_admin.delete_app(app)


@cli.command()
@click.argument("email")
def get_id_token(email: str) -> None:
    """Gets the id token of a user given their email."""

    API_KEY = "AIzaSyAMJgYSG_TW7CT_krdWaFUBLxU4yRINxX8"
    app = init_firebase()
    uid = auth.get_user_by_email(email).uid
    custom_token = auth.create_custom_token(uid)
    response = requests.post(
        "https://identitytoolkit.googleapis.com/v1/accounts:signInWithCustomToken?key="
        + API_KEY,
        data={"token": custom_token, "returnSecureToken": True},
    )
    click.echo(response.json().get("idToken"))
    firebase_admin.delete_app(app)


if __name__ == "__main__":
    cli()
