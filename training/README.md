## Run docker
From the training/ directory
Dev: docker-compose up
Production: docker-compose -f docker-compose.prod.yml up

## Rebuild and run docker
From the training/ directory
Dev: docker-compose up --build
Production: docker-compose -f docker-compose.prod.yml up --build