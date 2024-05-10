resource "aws_sqs_queue" "training_queue" {
  name       = "training-queue.fifo"
  fifo_queue = true
  message_retention_seconds = 60*24

  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.training_queue_deadletter.arn
    maxReceiveCount     = 4
  })
}

resource "aws_sqs_queue" "training_queue_deadletter" {
  name = "training-deadletter-queue"
}

resource "aws_sqs_queue_redrive_allow_policy" "training_queue_redrive_allow_policy" {
  queue_url = aws_sqs_queue.training_queue_deadletter.id

  redrive_allow_policy = jsonencode({
    redrivePermission = "byQueue",
    sourceQueueArns   = [aws_sqs_queue.training_queue.arn]
  })
}

output "sqs_queue_url" {
  value = aws_sqs_queue.training_queue.url
}