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
  region = "us-east-1"
}

resource "aws_dynamodb_table" "trainspace" {
  name           = "TrainspaceTable"
  billing_mode   = "PROVISIONED"
  hash_key       = "trainspace_id"
  write_capacity = 10
  read_capacity  = 10
  attribute {
    name = "trainspace_id"
    type = "S"
  }
  attribute {
    name = "user_id"
    type = "S"
  }
  global_secondary_index {
    name            = "user_id_index"
    hash_key        = "user_id"
    write_capacity  = 10
    read_capacity   = 10
    projection_type = "ALL"
  }
  point_in_time_recovery { enabled = true }
  server_side_encryption { enabled = true }
}

resource "aws_appautoscaling_target" "dynamodb_table_trainspace_read_target" {
  max_capacity       = 10
  min_capacity       = 1
  resource_id        = "table/${aws_dynamodb_table.trainspace.name}"
  scalable_dimension = "dynamodb:table:ReadCapacityUnits"
  service_namespace  = "dynamodb"
}
resource "aws_appautoscaling_policy" "dynamodb_table_trainspace_read_policy" {
  name               = "DynamoDBReadCapacityUtilization:${aws_appautoscaling_target.dynamodb_table_trainspace_read_target.resource_id}"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.dynamodb_table_trainspace_read_target.resource_id
  scalable_dimension = aws_appautoscaling_target.dynamodb_table_trainspace_read_target.scalable_dimension
  service_namespace  = aws_appautoscaling_target.dynamodb_table_trainspace_read_target.service_namespace

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "DynamoDBReadCapacityUtilization"
    }

    target_value = 70
  }
}
resource "aws_appautoscaling_target" "dynamodb_table_trainspace_write_target" {
  max_capacity       = 10
  min_capacity       = 1
  resource_id        = "table/${aws_dynamodb_table.trainspace.name}"
  scalable_dimension = "dynamodb:table:WriteCapacityUnits"
  service_namespace  = "dynamodb"
}
resource "aws_appautoscaling_policy" "dynamodb_table_trainspace_write_policy" {
  name               = "DynamoDBWriteCapacityUtilization:${aws_appautoscaling_target.dynamodb_table_trainspace_write_target.resource_id}"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.dynamodb_table_trainspace_write_target.resource_id
  scalable_dimension = aws_appautoscaling_target.dynamodb_table_trainspace_write_target.scalable_dimension
  service_namespace  = aws_appautoscaling_target.dynamodb_table_trainspace_write_target.service_namespace

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "DynamoDBWriteCapacityUtilization"
    }

    target_value = 70
  }
}


resource "aws_dynamodb_table" "user" {
  name           = "UserTable"
  billing_mode   = "PROVISIONED"
  hash_key       = "user_id"
  write_capacity = 10
  read_capacity  = 10
  attribute {
    name = "user_id"
    type = "S"
  }
  point_in_time_recovery { enabled = true }
  server_side_encryption { enabled = true }
}

resource "aws_appautoscaling_target" "dynamodb_table_user_read_target" {
  max_capacity       = 10
  min_capacity       = 1
  resource_id        = "table/${aws_dynamodb_table.user.name}"
  scalable_dimension = "dynamodb:table:ReadCapacityUnits"
  service_namespace  = "dynamodb"
}
resource "aws_appautoscaling_policy" "dynamodb_table_user_read_policy" {
  name               = "DynamoDBReadCapacityUtilization:${aws_appautoscaling_target.dynamodb_table_user_read_target.resource_id}"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.dynamodb_table_user_read_target.resource_id
  scalable_dimension = aws_appautoscaling_target.dynamodb_table_user_read_target.scalable_dimension
  service_namespace  = aws_appautoscaling_target.dynamodb_table_user_read_target.service_namespace

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "DynamoDBReadCapacityUtilization"
    }

    target_value = 70
  }
}
resource "aws_appautoscaling_target" "dynamodb_table_user_write_target" {
  max_capacity       = 10
  min_capacity       = 1
  resource_id        = "table/${aws_dynamodb_table.user.name}"
  scalable_dimension = "dynamodb:table:WriteCapacityUnits"
  service_namespace  = "dynamodb"
}
resource "aws_appautoscaling_policy" "dynamodb_table_user_write_policy" {
  name               = "DynamoDBWriteCapacityUtilization:${aws_appautoscaling_target.dynamodb_table_user_write_target.resource_id}"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.dynamodb_table_user_write_target.resource_id
  scalable_dimension = aws_appautoscaling_target.dynamodb_table_user_write_target.scalable_dimension
  service_namespace  = aws_appautoscaling_target.dynamodb_table_user_write_target.service_namespace

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "DynamoDBWriteCapacityUtilization"
    }

    target_value = 70
  }
}

resource "aws_dynamodb_table" "model" {
  name           = "ModelTable"
  billing_mode   = "PROVISIONED"
  hash_key       = "model_id"
  write_capacity = 10
  read_capacity  = 10
  attribute {
    name = "model_id"
    type = "S"
  }
  attribute {
    name = "user_id"
    type = "S"
  }
  global_secondary_index {
    name            = "user_id_index"
    hash_key        = "user_id"
    write_capacity  = 10
    read_capacity   = 10
    projection_type = "ALL"
  }
  point_in_time_recovery { enabled = true }
  server_side_encryption { enabled = true }
}

resource "aws_appautoscaling_target" "dynamodb_table_model_read_target" {
  max_capacity       = 10
  min_capacity       = 1
  resource_id        = "table/${aws_dynamodb_table.model.name}"
  scalable_dimension = "dynamodb:table:ReadCapacityUnits"
  service_namespace  = "dynamodb"
}
resource "aws_appautoscaling_policy" "dynamodb_table_model_read_policy" {
  name               = "DynamoDBReadCapacityUtilization:${aws_appautoscaling_target.dynamodb_table_model_read_target.resource_id}"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.dynamodb_table_model_read_target.resource_id
  scalable_dimension = aws_appautoscaling_target.dynamodb_table_model_read_target.scalable_dimension
  service_namespace  = aws_appautoscaling_target.dynamodb_table_model_read_target.service_namespace

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "DynamoDBReadCapacityUtilization"
    }

    target_value = 70
  }
}
resource "aws_appautoscaling_target" "dynamodb_table_model_write_target" {
  max_capacity       = 10
  min_capacity       = 1
  resource_id        = "table/${aws_dynamodb_table.model.name}"
  scalable_dimension = "dynamodb:table:WriteCapacityUnits"
  service_namespace  = "dynamodb"
}
resource "aws_appautoscaling_policy" "dynamodb_table_model_write_policy" {
  name               = "DynamoDBWriteCapacityUtilization:${aws_appautoscaling_target.dynamodb_table_model_write_target.resource_id}"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.dynamodb_table_model_write_target.resource_id
  scalable_dimension = aws_appautoscaling_target.dynamodb_table_model_write_target.scalable_dimension
  service_namespace  = aws_appautoscaling_target.dynamodb_table_model_write_target.service_namespace

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "DynamoDBWriteCapacityUtilization"
    }

    target_value = 70
  }
}