import pytest
import csv
import boto3
from botocore.exceptions import ClientError
from io import StringIO
from moto import mock_s3
from common.constants import *
from common.dataset import read_dataset

# Define the S3 bucket name
S3_BUCKET_NAME = "dlp-upload-bucket"
S3_REGION = "us-west-2"


@pytest.fixture
def s3_client():
    with mock_s3():
        s3 = boto3.client("s3", region_name=S3_REGION)
        s3.create_bucket(
            Bucket=S3_BUCKET_NAME,
            CreateBucketConfiguration={"LocationConstraint": S3_REGION},
        )
        yield s3


@pytest.mark.parametrize(
    "url,path_to_file",
    [
        (
            "https://raw.githubusercontent.com/karkir0003/dummy/main/job.csv",
            "./tests/expected/job.csv",
        ),
        (
            "https://raw.githubusercontent.com/karkir0003/dummy/main/cars.csv",
            "./tests/expected/cars.csv",
        ),
    ],
)
def test_dataset_url_reading(url, path_to_file, s3_client):
    try:
        # Call read_dataset, which should upload the CSV to the mock S3 bucket
        read_dataset(url)

        # Download the CSV data from the mock S3 bucket
        s3_response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key="data.csv")
        s3_data = s3_response["Body"].read().decode("utf-8")

        with open(path_to_file) as f:
            expected_data = f.read()

        # Compare the content of the CSV file uploaded to the mock S3 bucket with the expected CSV file
        assert s3_data == expected_data

    except Exception as e:
        pytest.fail(str(e))


@pytest.mark.parametrize(
    "url,path_to_file",
    [
        ("csv.com/jobs.csv", "./tests/expected/jobs.csv"),
        ("csv.com/cars.csv", "./tests/expected/cars.csv"),
    ],
)
def test_dataset_invalid_url(url, path_to_file, s3_client):
    with pytest.raises(Exception):
        read_dataset(url)
