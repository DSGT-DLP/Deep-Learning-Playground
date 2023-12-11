# pull base image
FROM condaforge/miniforge-pypy3
ARG ENV=dev
ENV NODE_ENV=${ENV}

WORKDIR /usr/src/training

# set environment variables

# install dependencies
RUN mamba create --name dlp -y

COPY environment.yml pyproject.toml poetry.lock ./
RUN mamba run --live-stream -n dlp mamba env update --file environment.yml --prune
RUN if [ "$NODE_ENV" = "dev" ]; then \
      mamba run --live-stream -n dlp poetry install; \
    else \
      mamba run --live-stream -n dlp poetry install --without dev; \
    fi

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