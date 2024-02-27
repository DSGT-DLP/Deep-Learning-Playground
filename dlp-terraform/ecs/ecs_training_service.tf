resource "aws_ecs_task_definition" "training" {
  family             = "training"
  task_role_arn      = aws_iam_role.ecs_task_role.arn
  execution_role_arn = aws_iam_role.ecs_exec_role.arn
  network_mode       = "bridge"
  cpu                = 1024
  memory             = 4096

  container_definitions = jsonencode([
    {
      "name" : "training",
      "image" : "${aws_ecr_repository.training.repository_url}:latest",
      "portMappings" : [
        {
          "name" : "gunicorn-port",
          "containerPort" : 8000,
          "hostPort" : 0,
          "protocol" : "tcp",
          "appProtocol" : "http"
        }
      ],
      "essential" : true,
      "environment" : [],
      "mountPoints" : [],
      "volumesFrom" : [],
      "logConfiguration" : {
        "logDriver" : "awslogs",
        "options" : {
          "awslogs-create-group" : "true",
          "awslogs-region" : "us-east-1",
          "awslogs-group" : aws_cloudwatch_log_group.training.name,
          "awslogs-stream-prefix" : "ecs"
        }
      }
    }
  ])
}

# --- ECS Service ---
resource "aws_ecs_service" "training" {
  name            = "training"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.training.arn
  desired_count   = 2

  capacity_provider_strategy {
    capacity_provider = aws_ecs_capacity_provider.training.name
    base              = 1
    weight            = 100
  }

  ordered_placement_strategy {
    type  = "spread"
    field = "attribute:ecs.availability-zone"
  }

  lifecycle {
    ignore_changes = [desired_count]
  }

  # load_balancer {
  #   target_group_arn = aws_lb_target_group.app.arn
  #   container_name = "training"
  #   container_port = 8000
  # }

  # depends_on = [aws_lb_target_group.app]
}

# --- ECS Service Auto Scaling ---
resource "aws_appautoscaling_target" "training_ecs_target" {
  service_namespace = "ecs"
  scalable_dimension = "ecs:service:DesiredCount"
  resource_id = "service/${aws_ecs_cluster.main.name}/${aws_ecs_service.training.name}"
  min_capacity = 0
  max_capacity = 2
}

resource "aws_appautoscaling_policy" "training_ecs_target_cpu" {
  name = "training-application-scaling-policy-cpu"
  policy_type = "TargetTrackingScaling"
  service_namespace = aws_appautoscaling_target.training_ecs_target.service_namespace
  resource_id = aws_appautoscaling_target.training_ecs_target.resource_id
  scalable_dimension = aws_appautoscaling_target.training_ecs_target.scalable_dimension

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }

    target_value = 80
    scale_in_cooldown = 300
    scale_out_cooldown = 300
  }
}

resource "aws_appautoscaling_policy" "training_ecs_target_memory" {
  name = "training-application-scaling-policy-memory"
  policy_type = "TargetTrackingScaling"
  service_namespace = aws_appautoscaling_target.training_ecs_target.service_namespace
  resource_id = aws_appautoscaling_target.training_ecs_target.resource_id
  scalable_dimension = aws_appautoscaling_target.training_ecs_target.scalable_dimension

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageMemoryUtilization"
    }

    target_value = 80
    scale_in_cooldown = 300
    scale_out_cooldown = 300
  }
}

# --- ECS Training Security Group ---
resource "aws_security_group" "ecs_training_sg" {
  name_prefix = "backend-ecs-training-sg-"
  vpc_id      = aws_vpc.main.id
}

resource "aws_vpc_security_group_ingress_rule" "ecs_training_sg_ingress" {
  security_group_id = aws_security_group.ecs_training_sg.id

  ip_protocol    = "-1"
  # cidr_blocks = [aws_vpc.main.cidr_block]
  referenced_security_group_id = aws_security_group.ecs_django_sg.id
} 

resource "aws_vpc_security_group_egress_rule" "ecs_training_sg_egress" {
  security_group_id = aws_security_group.ecs_training_sg.id

  ip_protocol    = "-1"
  cidr_ipv4 = "0.0.0.0/0"
} 

# --- ECS Launch Template ---
resource "aws_launch_template" "ecs_lt_training" {
  name_prefix   = "training-ecs-template-"
  image_id      = "ami-01ff5874b57a57613"
  instance_type = "g4dn.xlarge"

  vpc_security_group_ids = [aws_security_group.ecs_training_sg.id]
  iam_instance_profile {
    arn = aws_iam_instance_profile.ecs_node.arn
  }
  monitoring {
    enabled = true
  }

  user_data = base64encode(<<-EOF
      #!/bin/bash
      echo ECS_CLUSTER=${aws_ecs_cluster.main.name} >> /etc/ecs/ecs.config;
    EOF
  )
}

# --- ECS ASG ---
resource "aws_autoscaling_group" "training" {
  name_prefix               = "training-ecs-asg-"
  vpc_zone_identifier       = aws_subnet.public[*].id
  min_size                  = 0
  max_size                  = 1
  desired_capacity          = 1
  health_check_grace_period = 0
  health_check_type         = "EC2"
  protect_from_scale_in     = false

  launch_template {
    id      = aws_launch_template.ecs_lt_training.id
    version = "$Latest"
  }

  tag {
    key                 = "Name"
    value               = "backend-ecs-cluster"
    propagate_at_launch = true
  }

  tag {
    key                 = "AmazonECSManaged"
    value               = ""
    propagate_at_launch = true
  }
}

# --- ECS Capacity Provider ---
resource "aws_ecs_capacity_provider" "training" {
  name = "training-ecs-ec2"

  auto_scaling_group_provider {
    auto_scaling_group_arn         = aws_autoscaling_group.training.arn
    managed_termination_protection = "DISABLED"

    managed_scaling {
      maximum_scaling_step_size = 2
      minimum_scaling_step_size = 1
      status                    = "ENABLED"
      target_capacity           = 100
    }
  }
}

resource "aws_ecs_cluster_capacity_providers" "main" {
  cluster_name       = aws_ecs_cluster.main.name
  capacity_providers = [aws_ecs_capacity_provider.training.name]

  default_capacity_provider_strategy {
    capacity_provider = aws_ecs_capacity_provider.training.name
    base              = 1
    weight            = 100
  }
}