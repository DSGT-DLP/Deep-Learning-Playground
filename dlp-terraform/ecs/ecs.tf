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

resource "aws_ecs_cluster" "deep-learning-playground-kernels" {
  name = "deep-learning-playground-kernels-test"
  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}
resource "aws_ecs_service" "dlp-training-service" {
  name            = "dlp-training-service-test"
  cluster         = aws_ecs_cluster.deep-learning-playground-kernels.id
  task_definition = "arn:aws:ecs:us-west-2:521654603461:task-definition/dlp-training-task:9"
  desired_count   = 1

  launch_type = "FARGATE"

  deployment_maximum_percent         = "200"
  deployment_minimum_healthy_percent = "100"
  scheduling_strategy                = "REPLICA"

  network_configuration {
    security_groups  = ["sg-09291eb84a19daeed"]
    subnets          = ["subnet-0bebe768ad78b896c", "subnet-0f3e41ad21cfe6ff5"]
    assign_public_ip = true
  }
}
resource "aws_appautoscaling_target" "dev_to_target" {
  max_capacity       = 1
  min_capacity       = 1
  resource_id        = "service/${aws_ecs_cluster.deep-learning-playground-kernels.name}/${aws_ecs_service.dlp-training-service.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}
resource "aws_appautoscaling_policy" "training_service_auto_scaling_policy" {
  name               = "TrainingServiceAutoScalingPolicy"
  policy_type        = "StepScaling"
  resource_id        = "service/${aws_ecs_cluster.deep-learning-playground-kernels.name}/${aws_ecs_service.dlp-training-service.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"

  step_scaling_policy_configuration {
    adjustment_type         = "ChangeInCapacity"
    cooldown                = 30
    metric_aggregation_type = "Average"

    step_adjustment {
      metric_interval_lower_bound = 0
      scaling_adjustment          = 3
    }
  }

  depends_on = [
    aws_appautoscaling_target.dev_to_target
  ]
}
resource "aws_appautoscaling_policy" "dlp-queue-size-too-small-policy" {
  name               = "DLPQueueSizeTooSmallPolicy"
  policy_type        = "StepScaling"
  resource_id        = "service/${aws_ecs_cluster.deep-learning-playground-kernels.name}/${aws_ecs_service.dlp-training-service.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"

  step_scaling_policy_configuration {
    adjustment_type         = "ExactCapacity"
    cooldown                = 30
    metric_aggregation_type = "Average"

    step_adjustment {

      metric_interval_upper_bound = 0 
      scaling_adjustment          = 1
    }
  }
  depends_on = [aws_appautoscaling_target.dev_to_target]
}
