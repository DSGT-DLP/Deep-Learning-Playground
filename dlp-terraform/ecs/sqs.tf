resource "aws_sqs_queue" "training_queue" {
  name       = "training-queue.fifo"
  fifo_queue = true
  message_retention_seconds = 60*24
}

output "sqs_queue_url" {
  value = aws_sqs_queue.training_queue.url
}