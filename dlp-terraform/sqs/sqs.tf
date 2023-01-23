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

resource "aws_sqs_queue" "dlp_training_queue" {
  name                      = "dlp-training-queue-ter"
  delay_seconds             = 0
  max_message_size          = 262144
  message_retention_seconds = 345600
  receive_wait_time_seconds = 10
  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.dlp_dead_letter_queue.arn
    maxReceiveCount     = 10
  })

  tags = {
    Environment = "production"
  }
}

resource "aws_sqs_queue_policy" "dlp_sqs_queue_policy" {
  queue_url = aws_sqs_queue.dlp_training_queue.id

  policy = <<POLICY
{
 "Version": "2008-10-17",
  "Id": "__default_policy_ID",
  "Statement": [
    {
      "Sid": "__owner_statement",
      "Effect": "Allow",
      "Principal": {
        "AWS": "arn:aws:iam::521654603461:root"
      },
      "Action": "SQS:*",
      "Resource": "arn:aws:sqs:us-west-2:521654603461:dlp-training-queue"
    },
    {
      "Sid": "__sender_statement",
      "Effect": "Allow",
      "Principal": {
        "AWS": [
          "arn:aws:iam::521654603461:root",
          "arn:aws:iam::521654603461:role/ecsTaskExecutionRole"
        ]
      },
      "Action": "SQS:SendMessage",
      "Resource": "arn:aws:sqs:us-west-2:521654603461:dlp-training-queue"
    },
    {
      "Sid": "__receiver_statement",
      "Effect": "Allow",
      "Principal": {
        "AWS": [
          "arn:aws:iam::521654603461:root",
          "arn:aws:iam::521654603461:role/ecsTaskExecutionRole"
        ]
      },
      "Action": [
        "SQS:ChangeMessageVisibility",
        "SQS:DeleteMessage",
        "SQS:ReceiveMessage"
      ],
      "Resource": "arn:aws:sqs:us-west-2:521654603461:dlp-training-queue"
    }
  ]
}
POLICY
}

resource "aws_sqs_queue" "dlp_dead_letter_queue" {
  name = "dlp-deadletter-queue-ter"
  message_retention_seconds = 345600
  visibility_timeout_seconds = 30
  receive_wait_time_seconds = 10
}

resource "aws_sqs_queue_policy" "dlp_dead_letter_queue_policy" {
  queue_url = aws_sqs_queue.dlp_dead_letter_queue.id

  policy = <<POLICY
{
  "Version": "2008-10-17",
  "Id": "__default_policy_ID",
  "Statement": [
    {
      "Sid": "__owner_statement",
      "Effect": "Allow",
      "Principal": {
        "AWS": "arn:aws:iam::521654603461:root"
      },
      "Action": "SQS:*",
      "Resource": "arn:aws:sqs:us-west-2:521654603461:dead-letter-queue"
    },
    {
      "Sid": "__sender_statement",
      "Effect": "Allow",
      "Principal": {
        "AWS": [
          "arn:aws:iam::521654603461:root",
          "arn:aws:iam::521654603461:role/ecsTaskExecutionRole"
        ]
      },
      "Action": "SQS:SendMessage",
      "Resource": "arn:aws:sqs:us-west-2:521654603461:dead-letter-queue"
    },
    {
      "Sid": "__receiver_statement",
      "Effect": "Allow",
      "Principal": {
        "AWS": [
          "arn:aws:iam::521654603461:root",
          "arn:aws:iam::521654603461:role/ecsTaskExecutionRole"
        ]
      },
      "Action": [
        "SQS:ChangeMessageVisibility",
        "SQS:DeleteMessage",
        "SQS:ReceiveMessage"
      ],
      "Resource": "arn:aws:sqs:us-west-2:521654603461:dead-letter-queue"
    }
  ]
}
POLICY
}
