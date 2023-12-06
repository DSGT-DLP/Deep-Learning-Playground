# pull base image
FROM python:3.9-slim-bookworm
FROM condaforge/miniforge-pypy3

WORKDIR /usr/src/training

# set environment variables

# install dependencies
RUN mamba create --name dlp -y

COPY environment.yml pyproject.toml ./
RUN mamba run --live-stream -n dlp mamba env update --file environment.yml --prune
RUN mamba run --live-stream -n dlp poetry install

# copy project
COPY . .