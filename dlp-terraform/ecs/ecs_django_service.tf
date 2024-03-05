resource "aws_ecs_task_definition" "django" {
  family = "django"
  task_role_arn      = aws_iam_role.ecs_task_role.arn
  execution_role_arn = aws_iam_role.ecs_exec_role.arn
  network_mode = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                = 1024
  memory             = 2048

  container_definitions = jsonencode([
    {
      "name": "django",
      "image" : "${aws_ecr_repository.django.repository_url}:latest",
      "cpu": 1024,
      "memory": 2048,
      "essential": true,
      "portMappings": [
        {
          "name" : "gunicorn-port",
          "containerPort" : 8000,
          "hostPort" : 8000,
          "protocol" : "tcp",
        }
      ],
      "logConfiguration" : {
        "logDriver" : "awslogs",
        "options" : {
          "awslogs-create-group" : "true",
          "awslogs-region" : "us-east-1",
          "awslogs-group" : aws_cloudwatch_log_group.django.name,
          "awslogs-stream-prefix" : "ecs"
        }
      },
      "environment": [
        {
          "name": "ALLOWED_HOST",
          "value": "${aws_lb.main.dns_name}"
        }
      ]
    }
  ])
}

# --- ECS Django Security Group ---
resource "aws_security_group" "ecs_django_sg" {
  name_prefix = "backend-ecs-django-sg-"
  vpc_id      = aws_vpc.main.id
}

resource "aws_vpc_security_group_ingress_rule" "ecs_django_sg_ingress" {
  security_group_id = aws_security_group.ecs_django_sg.id

  ip_protocol    = "-1"
  referenced_security_group_id = aws_security_group.http.id
} 

resource "aws_vpc_security_group_egress_rule" "ecs_django_sg_egress" {
  security_group_id = aws_security_group.ecs_django_sg.id

  ip_protocol    = "-1"
  cidr_ipv4 = "0.0.0.0/0"
} 

resource "aws_ecs_service" "django" {
  name            = "django"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.django.arn
  desired_count   = 2
  launch_type = "FARGATE"

  network_configuration {
    security_groups = [ aws_security_group.ecs_django_sg.id]
    subnets = aws_subnet.public[*].id
    assign_public_ip = true
  }

  lifecycle {
    ignore_changes = [desired_count]
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.app.arn
    container_name = "django"
    container_port = 8000
  }

  depends_on = [aws_lb_target_group.app]
}