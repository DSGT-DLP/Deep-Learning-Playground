import boto3
import os
import shutil
from backend.common.constants import AWS_REGION

"""
This file contains wrappers to interface with S3 buckets through 
operations such as listing buckets, reading data from buckets, writing to buckets
"""

s3 = boto3.client('s3', region_name=AWS_REGION)

def generate_presigned_post(bucket_name: str, bucket_path: str, expiration: int = 300):
    """
    Given a bucket name, bucket path, and expiration time (in seconds), generate a presigned URL
    for the given bucket path

    Args:
        bucket_name (str): name of s3 bucket
        bucket_path (str): path within s3 bucket where the file resides
        expiration (int): expiration time in seconds (default is 300 seconds or 5 minutes)
    """
    return s3.generate_presigned_post(Bucket=bucket_name, Key=bucket_path, ExpiresIn=expiration)

def write_to_bucket(file_path: str, bucket_name: str, bucket_path: str):
    """
    Given a file path and a location in s3, write the file to that location

    Args:
        file_path (str): path to file
        bucket_name (str): name of s3 bucket
        bucket_path (str): path within s3 bucket where the file should live  
    
    S3 URIs are formatted as such "s3://<bucket_name>/<path_to_file>"      
    """
    s3.meta.client.upload_file(Filename=file_path, Bucket=bucket_name, Key=bucket_path)

def read_from_bucket(bucket_name: str, bucket_path: str, output_file_name: str, output_file_path: str):
    """
    Given S3 URI, read the file from the S3 bucket

    Args:
        bucket_name (str): name of s3 bucket
        bucket_path (str): path within s3 bucket where the file resides
        output_file_name (str): name of file to download S3 object to (this is because of the way the boto3 endpoint works)
        output_file_path (str): filepath to download file to (ie: what folder/directory)
        
    """
    if (not os.path.exists(output_file_path)):
        os.makedirs(output_file_path)
    s3.Bucket(bucket_name).download_file(bucket_path, output_file_name)
    shutil.move(f"{os.getcwd()}/{output_file_name}", f"{output_file_path}/{output_file_name}")


