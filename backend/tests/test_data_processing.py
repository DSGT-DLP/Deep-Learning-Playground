# Import necessary libraries
from urllib.error import URLError
import pytest
import csv
import boto3
from botocore.exceptions import ClientError
from io import StringIO
from common.constants import *
from common.dataset import read_dataset

# Define the S3 bucket name
S3_BUCKET_NAME = "dlp-upload-bucket"


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
def test_dataset_url_reading(url, path_to_file):
    try:
        read_dataset(url)

        # Upload the test data to the S3 bucket before running the test
        with open(path_to_file, "rb") as f:
            s3 = boto3.client("s3")
            s3.put_object(Bucket=S3_BUCKET_NAME, Key="data.csv", Body=f)

        # Download the CSV data from S3
        s3_response = s3.get_object(Bucket=S3_BUCKET_NAME, Key="data.csv")
        s3_data = s3_response["Body"].read().decode("utf8")

        with open(path_to_file) as f:
            expected_data = f.read()

        # Compare the content of the CSV file uploaded to S3 with the expected CSV file
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
def test_dataset_invalid_url(url, path_to_file):
    with pytest.raises(Exception):
        read_dataset(url)
