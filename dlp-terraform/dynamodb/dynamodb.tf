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

resource "aws_dynamodb_table" "dlp-file-upload-table" {
  name           = "dlp-file-upload-table"
  hash_key       = "s3_uri"
  billing_mode   = "PROVISIONED"
  write_capacity = 10
  read_capacity  = 10
  attribute {
    name = "s3_uri"
    type = "S"
  }
  attribute {
    name = "uid"
    type = "S"
  }
  ttl {
    enabled        = true
    attribute_name = "ttl"
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
