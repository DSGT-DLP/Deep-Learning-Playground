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

# create directory for the app user
RUN mkdir -p /home/app

# create the app user
RUN addgroup --system app && adduser --system --group app

# create the appropriate directories
ENV HOME=/home/app
ENV APP_HOME=/home/app/web
RUN mkdir $APP_HOME
WORKDIR $APP_HOME

# copy project
COPY . $APP_HOME

# chown all the files to the app user
RUN chown -R app:app $APP_HOME

# change to the app user
USER app