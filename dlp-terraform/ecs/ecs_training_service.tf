resource "aws_ecs_task_definition" "training" {
  family             = "training"
  task_role_arn      = aws_iam_role.ecs_task_role.arn
  execution_role_arn = aws_iam_role.ecs_exec_role.arn
  network_mode       = "bridge"
  cpu                = 1024
  memory             = 4096

  container_definitions = jsonencode(([
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
          "awslogs-group" : aws_cloudwatch_log_group.ecs.name,
          "awslogs-stream-prefix" : "ecs"
        }
      }
    }
  ]))
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

  load_balancer {
    target_group_arn = aws_lb_target_group.app.arn
    container_name = "training"
    container_port = 8000
  }

  depends_on = [aws_lb_target_group.app]
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