resource "aws_ecr_repository" "training" {
  name                 = "training"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}

output "training_repo_url" {
  value = aws_ecr_repository.training.repository_url
}
