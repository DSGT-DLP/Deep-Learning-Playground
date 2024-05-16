## Run docker
Ensure you have logged in to AWS, using (`aws configure sso`) then (`aws sso login --profile=dlp`)

From the training/ directory
Dev: `AWS_PROFILE=dlp docker-compose up`
Production: `AWS_PROFILE=dlp docker compose -f docker-compose.prod.yml up`

To rebuild, add a --build flag to the command
