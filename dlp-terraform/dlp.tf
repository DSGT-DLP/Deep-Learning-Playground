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

resource "aws_s3_bucket" "s3bucket" {
  bucket = "asdfasga-hello-tacocat"

  tags = {
    Name        = "My bucket"
    Environment = "Dev"
  }

}

resource "aws_s3_bucket_acl" "example" {
  bucket = aws_s3_bucket.s3bucket.id
  acl    = "private"
}
