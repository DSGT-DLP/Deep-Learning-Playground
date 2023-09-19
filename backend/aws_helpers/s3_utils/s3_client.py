from aws_helpers.dynamo_db_utils.trainspace_db import TrainspaceData
import boto3
import os
import shutil
import pandas as pd
import io

from aws_helpers.s3_utils.s3_bucket_names import FILE_UPLOAD_BUCKET_NAME

"""
This file contains wrappers to interface with S3 buckets through 
operations such as listing buckets, reading data from buckets, writing to buckets
"""


def write_to_bucket(file_path: str, bucket_name: str, bucket_path: str):
    """
    Given a file path and a location in s3, write the file to that location

    Args:
        file_path (str): path to file
        bucket_name (str): name of s3 bucket
        bucket_path (str): path within s3 bucket where the file should live

    S3 URIs are formatted as such "s3://<bucket_name>/<path_to_file>"
    """
    s3 = boto3.resource("s3")
    s3.meta.client.upload_file(Filename=file_path, Bucket=bucket_name, Key=bucket_path)


def read_from_bucket(
    bucket_name: str, bucket_path: str, output_file_name: str, output_file_path: str
) -> None:
    """
    Given S3 URI, read the file from the S3 bucket

    Args:
        bucket_name (str): name of s3 bucket
        bucket_path (str): path within s3 bucket where the file resides
        output_file_name (str): name of file to download S3 object to (this is because of the way the boto3 endpoint works)
        output_file_path (str): filepath to download file to (ie: what folder/directory)

    """
    s3 = boto3.resource("s3")
    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path)
    s3.Bucket(bucket_name).download_file(bucket_path, output_file_name)
    shutil.move(
        f"{os.getcwd()}/{output_file_name}", f"{output_file_path}/{output_file_name}"
    )


def read_df_from_bucket(bucket_name: str, bucket_path: str) -> pd.DataFrame:
    """
    Given S3 URI, read the file from the S3 bucket and return a pandas dataframe

    Args:
        bucket_name (str): name of s3 bucket
        bucket_path (str): path within s3 bucket where the file resides

    """
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket_name, Key=bucket_path)
    df = pd.read_csv(io.BytesIO(obj["Body"].read()))
    return df


def make_train_bucket_path(trainspace_data: TrainspaceData) -> str:
    """
    Given a TrainspaceData object, return the path to the bucket where the training data will be stored

    Args:
        trainspace_data (TrainspaceData): object containing data about the training data

    Returns:
        bucket_path (str): path to bucket where training data will be stored
    """
    uid = trainspace_data.uid
    data_source = trainspace_data.data_source.lower()
    filename = trainspace_data.dataset_data["name"]
    return f"{uid}/{data_source}/{filename}"


def get_presigned_url_from_bucket(bucket_name: str, bucket_path: str):
    """
    Given S3 URI, read the file from the S3 bucket

    Args:
        bucket_name (str): name of s3 bucket
        bucket_path (str): path within s3 bucket where the file resides
        output_file_name (str): name of file to download S3 object to (this is because of the way the boto3 endpoint works)
        output_file_path (str): filepath to download file to (ie: what folder/directory)

    """
    s3 = boto3.client("s3")
    return s3.generate_presigned_url(
        "get_object", Params={"Bucket": bucket_name, "Key": bucket_path}
    )


def get_presigned_upload_post_from_bucket(bucket_name: str, object_name: str):
    """
    Generate a presigned URL to upload a file to an S3 bucket

    Args:
        bucket_name (str): The name of the S3 bucket.
        object_name (str): The name of the object to upload.
    """
    s3 = boto3.client("s3")
    return s3.generate_presigned_post(bucket_name, object_name)


def get_objects_in_folder(bucket_name: str, folder_prefix: str):
    """
    Get the object data for all objects in a specified folder in an S3 bucket.

    Args:
        bucket_name (str): The name of the S3 bucket.
        folder_prefix (str): The prefix of the folder in the S3 bucket.

    Returns:
        A list of dictionaries containing data for each object in the folder.
    """
    # Create an S3 client
    s3 = boto3.client("s3")

    # List all objects in the specified folder
    objects = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_prefix)
    if not "Contents" in objects:
        return []
    return objects["Contents"]


def get_user_dataset_file_objects(user_id: str, data_source: str):
    """
    Get the object data for all objects stored by a user in an S3 bucket.

    Args:
        user_id (str): The id of the user.
        data_source (str): The data source of the objects.

    Returns:
        A list of dictionaries containing data for each object stored by the user.
    """
    return get_objects_in_folder(FILE_UPLOAD_BUCKET_NAME, f"{user_id}/{data_source}")


def get_presigned_url_from_exec_file(bucket_name: str, exec_id: str, filename: str):
    return get_presigned_url_from_bucket(bucket_name, exec_id + "/" + filename)


def get_presigned_upload_post_from_user_dataset_file(
    user_id: str, data_source: str, filename: str
):
    """
    Get the presigned url for a file stored by a user in the file upload S3 bucket.

    Args:
        user_id (str): The id of the user.
        data_source (str): The data source of the object.
        filename (str): The name of the file.

    Returns:
        A dictionary containing a boto3 presigned url post object for the file.
    """
    post_obj = get_presigned_upload_post_from_bucket(
        FILE_UPLOAD_BUCKET_NAME, f"{user_id}/{data_source}/{filename}"
    )
    return post_obj


def get_column_names(user_id: str, data_source: str, filename: str):
    s3 = boto3.client("s3")
    response = s3.select_object_content(
        Bucket=FILE_UPLOAD_BUCKET_NAME,
        Key=f"{user_id}/{data_source}/{filename}",
        ExpressionType="SQL",
        Expression="SELECT * FROM S3Object LIMIT 1",
        InputSerialization={"CSV": {"FileHeaderInfo": "NONE"}},
        OutputSerialization={"CSV": {}},
    )
    # Iterate over the response records to extract the header row
    for event in response["Payload"]:
        if "Records" in event:
            records = event["Records"]["Payload"].decode("utf-8")
            print(records)
            columns = records.split("\n")[0].split(",")
    return columns
