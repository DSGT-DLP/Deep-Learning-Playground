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

resource "aws_dynamodb_table" "execution-table" {
  name           = "execution-table"
  hash_key       = "execution_id"
  billing_mode   = "PROVISIONED"
  write_capacity = 10
  read_capacity  = 10
  attribute {
    name = "execution_id"
    type = "S"
  }
  attribute {
    name = "user_id"
    type = "S"
  }
  ttl {
    enabled        = true
    attribute_name = "expiryPeriod"
  }
  global_secondary_index {
    name            = "user_id"
    hash_key        = "user_id"
    write_capacity  = 10
    read_capacity   = 10
    projection_type = "ALL"
  }
  point_in_time_recovery { enabled = true }
  server_side_encryption { enabled = true }
}
resource "aws_dynamodb_table" "userprogress_table" {
  name           = "userprogress_table"
  hash_key       = "uid"
  billing_mode   = "PROVISIONED"
  write_capacity = 1
  read_capacity  = 1
  lifecycle {
    ignore_changes = [read_capacity, write_capacity]
  }
  attribute {
    name = "uid"
    type = "S"
  }
  ttl {
    enabled        = true
    attribute_name = "expiryPeriod"
  }
  point_in_time_recovery { enabled = true }
  server_side_encryption { enabled = true }
}

resource "aws_dynamodb_table" "trainspace" {
  name           = "trainspace"
  hash_key       = "trainspace_id"
  billing_mode   = "PROVISIONED"
  write_capacity = 10
  read_capacity  = 10
  attribute {
    name = "trainspace_id"
    type = "S"
  }
  attribute {
    name = "uid"
    type = "S"
  }
  ttl {
    enabled        = true
    attribute_name = "expiryPeriod"
  }
  global_secondary_index {
    name            = "uid"
    hash_key        = "uid"
    write_capacity  = 10
    read_capacity   = 10
    projection_type = "ALL"
  }
  point_in_time_recovery { enabled = true }
  server_side_encryption { enabled = true }
}

resource "aws_appautoscaling_target" "dynamodb_table_userprogress_read_target" {
  max_capacity       = 10
  min_capacity       = 1
  resource_id        = "table/${aws_dynamodb_table.userprogress_table.name}"
  scalable_dimension = "dynamodb:table:ReadCapacityUnits"
  service_namespace  = "dynamodb"
}
resource "aws_appautoscaling_policy" "dynamodb_table_userprogress_read_policy" {
  name               = "DynamoDBReadCapacityUtilization:${aws_appautoscaling_target.dynamodb_table_userprogress_read_target.resource_id}"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.dynamodb_table_userprogress_read_target.resource_id
  scalable_dimension = aws_appautoscaling_target.dynamodb_table_userprogress_read_target.scalable_dimension
  service_namespace  = aws_appautoscaling_target.dynamodb_table_userprogress_read_target.service_namespace

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "DynamoDBReadCapacityUtilization"
    }

    target_value = 70
  }
}
resource "aws_appautoscaling_target" "dynamodb_table_userprogress_write_target" {
  max_capacity       = 10
  min_capacity       = 1
  resource_id        = "table/${aws_dynamodb_table.userprogress_table.name}"
  scalable_dimension = "dynamodb:table:WriteCapacityUnits"
  service_namespace  = "dynamodb"
}
resource "aws_appautoscaling_policy" "dynamodb_table_userprogress_write_policy" {
  name               = "DynamoDBWriteCapacityUtilization:${aws_appautoscaling_target.dynamodb_table_userprogress_write_target.resource_id}"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.dynamodb_table_userprogress_write_target.resource_id
  scalable_dimension = aws_appautoscaling_target.dynamodb_table_userprogress_write_target.scalable_dimension
  service_namespace  = aws_appautoscaling_target.dynamodb_table_userprogress_write_target.service_namespace

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "DynamoDBWriteCapacityUtilization"
    }

    target_value = 70
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
