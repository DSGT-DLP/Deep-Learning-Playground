terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.16"
    }
  }

  required_version = ">= 1.2.0"
}

provider "aws" {
  region = "us-west-2"
}

resource "aws_s3_bucket" "s3bucket_executions" {
  bucket = "dlp-executions-bucket"

  tags = {
    Name        = "Execution data"
    Environment = "Dev"
  }
}
resource "aws_s3_bucket" "s3bucket_uploads" {
  bucket = "dlp-upload-bucket"

  tags = {
    Name        = "Upload data"
    Environment = "Dev"
  }
}
resource "aws_s3_bucket_versioning" "upload_versioning" {
  bucket = aws_s3_bucket.s3bucket_uploads.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "upload_bucket_config" {
  bucket = aws_s3_bucket.s3bucket_uploads.bucket
  rule {
    expiration {
      days = 5
    }
    id     = "logs"
    status = "Enabled"
  }
}

resource "aws_s3_bucket_public_access_block" "access_block_executions" {
  bucket = aws_s3_bucket.s3bucket_executions.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
resource "aws_s3_bucket_public_access_block" "access_block_uploads" {
  bucket = aws_s3_bucket.s3bucket_uploads.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
resource "aws_s3_bucket_cors_configuration" "uploads_cors_configuration" {
  bucket = aws_s3_bucket.s3bucket_uploads.id

  cors_rule {
    allowed_headers = ["*"]
    allowed_methods = ["GET", "PUT", "POST", "HEAD", "DELETE"]
    allowed_origins = ["*"]
    expose_headers  = []
    max_age_seconds = 3000
  }
}
