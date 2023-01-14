import pytest
from backend.aws_helpers.lambda_utils.lambda_client import invoke
import json
import pandas as pd
from moto import mock_iam, mock_lambda
import boto3
import zipfile

@pytest.fixture
def read_csv_to_json():
    df = pd.read_csv("tests/test_data/iris.csv")
    return df.to_dict("records")

@mock_iam    
@mock_lambda
def invoke_preprocess_lambda(payload):
    session = boto3.Session(
      aws_access_key_id="fake_access_key",
      aws_secret_access_key="fake_secret_key",
      region_name="us-west-2"
    )
    iam_client = session.client("iam")
    role = iam_client.create_role(
        RoleName="preprocess_test_role",
        AssumeRolePolicyDocument=json.dumps({
            "Statement": [{
                "Action": "sts:AssumeRole",
                "Principal": {"Service": "lambda.amazonaws.com"},
                "Effect": "Allow"
            }]
        })
    )

    # Attach necessary permissions to the role
    iam_client.attach_role_policy(
        RoleName="preprocess_test_role",
        PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
    )
    
    # Get the ARN of the new role
    role_arn = role["Role"]["Arn"]
    
    client = session.client('lambda')
    client.create_function(
        FunctionName='preprocess_data',
        Runtime='python3.9',
        Role=role_arn,
        Handler='lambda_function.lambda_handler',
        Code={'ZipFile': open("dlp-terraform/lambda/preprocess_lambda_function.py", "rb").read()}
    )
    response = client.invoke(FunctionName="preprocess_data", Payload=payload)
    resJson = response["Payload"].read()
    print(resJson)
    return resJson
    
@pytest.mark.parametrize(
    "code_snippet",
    [
        """import pandas as pd\ndef preprocess(df):\n\tdf = df.rename(columns={"variety": "dlp"})\n\treturn df""",
    ],
)
def test_rename_column(read_csv_to_json, code_snippet):
    payload = json.dumps({
            "data": read_csv_to_json,
            "code": code_snippet
    })
    resJson = invoke_preprocess_lambda(payload)
    assert "columns" in resJson and resJson['columns'] == ['sepal.length', 'sepal.width', 'petal.length', 'petal.width', 'dlp'], resJson

@pytest.mark.parametrize(
    "code_snippet",
    [
        """import pandas as pd\ndef preprocess(df):\n\tdf = df.drop(columns=['variety'])\n\treturn df""",
    ],
)
def test_drop_column(read_csv_to_json, code_snippet):
    payload = json.dumps({
            "data": read_csv_to_json,
            "code": code_snippet
    })
    resJson = invoke_preprocess_lambda(payload)
    assert "columns" in resJson and resJson['columns'] == ['sepal.length', 'sepal.width', 'petal.length', 'petal.width'], resJson

@pytest.mark.parametrize(
    "code_snippet",
    [
        """import pandas as pd\ndef preprocess(df):\n\tdf['test'] = df['sepal.length'] + df['sepal.width']\n\treturn df""",
    ],
)
def test_add_two_columns(read_csv_to_json, code_snippet):
    payload = json.dumps({
            "data": read_csv_to_json,
            "code": code_snippet
    })
    resJson = invoke_preprocess_lambda(payload)
    assert "columns" in resJson and resJson['columns'] == ['sepal.length', 'sepal.width', 'petal.length', 'petal.width', 'variety', 'test'], resJson