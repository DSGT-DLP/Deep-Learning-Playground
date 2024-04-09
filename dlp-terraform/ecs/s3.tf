resource "aws_s3_bucket" "s3bucket_executions" {
  bucket = "dlp-executions"

  tags = {
    Name = "Execution data"
  }
}
resource "aws_s3_bucket_public_access_block" "access_block_uploads" {
  bucket = aws_s3_bucket.s3bucket_executions.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
