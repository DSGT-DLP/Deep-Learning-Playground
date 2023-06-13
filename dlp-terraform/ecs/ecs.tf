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

resource "aws_instance" "dlp-ec2-instance" {
  ami                    = "ami-07dfed28fcf95241c"
  instance_type          = "t2.micro"
  key_name               = "aws_key"
  vpc_security_group_ids = [aws_security_group.main.id]
  tags = {
    Name = "dlp-ec2-instance-test"
  }
}

resource "aws_security_group" "main" {
  egress = [
    {
      cidr_blocks      = ["0.0.0.0/0", ]
      description      = ""
      from_port        = 0
      ipv6_cidr_blocks = []
      prefix_list_ids  = []
      protocol         = "-1"
      security_groups  = []
      self             = false
      to_port          = 0
    }
  ]
  ingress = [
    {
      cidr_blocks      = ["0.0.0.0/0", ]
      description      = ""
      from_port        = 22
      ipv6_cidr_blocks = []
      prefix_list_ids  = []
      protocol         = "tcp"
      security_groups  = []
      self             = false
      to_port          = 22
    }
  ]
}


resource "aws_key_pair" "deployer" {
  key_name   = "aws_key"
  public_key = "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQDFPoGs5b17w2yuwZLUbIvj/nJvaPmqOxTxtlpqaWGwFtnO8uKPeyFAHPwLZn31jqJmgBFyCS/6wqetIZLZE7b9I+U9s8FyxvKs+kPqTvvTi47etN7mKLMiTBm8m+mSH5Knljli/C0bdN9CCWKRnoa4hU++hJspWOBKgvmY3TpT8UBJt2Ht9IDy1yzaoEGJ6c31aV4dehlzFcUCx4cV/b6AtigJCisBoXIrV4HF7XuJSYUnXPqbXD7s7lvCRFc6p1z/S2+E7fR4V76mQ1HbusU/gcU3sFZ/qs+tfu2mxx8YBz09vACO+d8eBBsRLSESGbhjKMDQD6lYzjTfa9uHYkbx noahi24@Noah-ASUS"
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
