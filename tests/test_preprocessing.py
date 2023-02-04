import pytest
from backend.aws_helpers.lambda_utils.lambda_client import invoke
import json
import pandas as pd
from moto import mock_iam, mock_lambda, mock_s3
import io
import zipfile
import boto3
import os 

os.environ["MOTO_DOCKER_LAMBDA_IMAGE"] = "mlupin/docker-lambda"
@pytest.fixture
def read_csv_to_json():
    df = pd.read_csv("tests/test_data/iris.csv")
    return df.to_dict("records")

def create_zip_file(file):
    zip_output = io.BytesIO()
    zip_file = zipfile.ZipFile(zip_output, "w", zipfile.ZIP_DEFLATED)
    zip_file.writestr("lambda_function.py", open(file, "rb").read())
    zip_file.close()
    zip_output.seek(0)
    return zip_output.read()

@mock_s3
@mock_iam    
def invoke_preprocess_lambda(payload):
    with mock_lambda():
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
        lambda_func = boto3.client('lambda', aws_access_key_id="fake_access_key",
                            aws_secret_access_key="fake_secret_key",
                            region_name='us-west-2')
        
        s3 = boto3.client('s3', aws_access_key_id="fake_access_key",
                            aws_secret_access_key="fake_secret_key",
                            region_name='us-west-2')
        s3.create_bucket(Bucket='pandas-bucket', CreateBucketConfiguration={'LocationConstraint': 'us-west-2'})
        pandas_layer = pd.__file__
        s3.upload_file(pandas_layer, 'pandas-bucket', 'pandas_layer.zip')
        
        pandas_lambda_layer = lambda_func.publish_layer_version(
            LayerName='pandas-layer',
            Content={
                'S3Bucket': 'pandas-bucket',
                'S3Key': 'pandas_layer.zip'
            },
            CompatibleRuntimes=['python3.0']
        )
        pandas_arn = pandas_lambda_layer['LayerVersionArn']
        
        lambda_func.create_function(
            FunctionName='preprocess_data',
            Runtime='python3.9',
            Role=role_arn,
            Handler='lambda_function.lambda_handler',
            Code={'ZipFile': create_zip_file("dlp-terraform/lambda/preprocess_lambda_function.py")},
            Layers=[pandas_arn]
        )
        response = lambda_func.invoke(FunctionName="preprocess_data", Payload=payload)
        print(response["Payload"].read())
        return response["Payload"].read()
    
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