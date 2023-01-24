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
data "archive_file" "send_email_files" {
  type        = "zip"
  output_path = "outputs/send_email.zip"
  source_dir  = "lambda/source-files"
}

resource "aws_lambda_function" "send_email" {
  filename      = "outputs/send_email.zip"
  function_name = "send_email"
  role          = "arn:aws:iam::521654603461:role/send_email"
  handler       = "lambda_function.lambda_handler"

  source_code_hash = filebase64sha256("outputs/send_email.zip")

  runtime = "python3.9"
}

resource "aws_lambda_function_url" "send_email_url" {
  function_name      = aws_lambda_function.send_email.function_name
  authorization_type = "AWS_IAM"
}

resource "aws_apigatewayv2_api" "send_email" {
  name = "send_email"
  protocol_type = "HTTP"
  
}
resource "aws_apigatewayv2_stage" "default" {
  api_id = aws_apigatewayv2_api.send_email.id
  name   = "default"
  auto_deploy = true
}

resource "aws_apigatewayv2_integration" "send_email_integration" {
  api_id = aws_apigatewayv2_api.send_email.id
  integration_uri = aws_lambda_function.send_email.invoke_arn
  integration_type = "AWS_PROXY"
  integration_method = "POST"
  payload_format_version = "2.0"
}

resource "aws_apigatewayv2_route" "set_send_email" {
  api_id = aws_apigatewayv2_api.send_email.id
  
  route_key = "POST /send_email"
  target = "integrations/${aws_apigatewayv2_integration.send_email_integration.id}"
}

resource "aws_lambda_permission" "api_gw_send_email" {
  statement_id = "AllowExecutionFromAPIGateway"
  action = "lambda:InvokeFunction"
  function_name = aws_lambda_function.send_email.function_name
  principal = "apigateway.amazonaws.com"
  
  source_arn = "${aws_apigatewayv2_api.send_email.execution_arn}/*/*/send_email"
}
data "archive_file" "preprocess_data_files" {
  type        = "zip"
  output_path = "outputs/preprocess_data.zip"
  source_file = "lambda/preprocess_lambda_function.py"
}

resource "aws_lambda_function" "preprocess_data" {
  filename      = "outputs/preprocess_data.zip"
  function_name = "preprocess_data"
  role          = "arn:aws:iam::521654603461:role/service-role/preprocess_data-role-9328ks4z"
  handler       = "preprocess_lambda_function.lambda_handler"
  layers = ["arn:aws:lambda:us-west-2:336392948345:layer:AWSSDKPandas-Python39:2", "arn:aws:lambda:us-west-2:770693421928:layer:Klayers-p39-numpy:9"]

  source_code_hash = filebase64sha256("outputs/preprocess_data.zip")
  runtime = "python3.9"
}

resource "aws_lambda_function_url" "preprocess_data_url" {
  function_name      = aws_lambda_function.preprocess_data.function_name
  authorization_type = "AWS_IAM"
}