FROM condaforge/miniforge3:24.1.2-0 AS base

WORKDIR /usr/src/build

# install dependencies
RUN mamba create --name dlp -y

COPY environment.yml pyproject.toml poetry.lock ./
RUN mamba run --live-stream -n dlp mamba env update --file environment.yml --prune

# create directory for the app user
RUN mkdir -p /home/app

# create the app user
RUN addgroup --system app && adduser --system --group app

# create the appropriate directories
ARG HOME=/home/app
ARG APP_HOME=/home/app/training
RUN mkdir $APP_HOME



######################## development target ######################## 
FROM base AS development
ENV ENVIRONMENT development

RUN mamba run --live-stream -n dlp poetry install


# we previously mounted training/ to APP_HOME
WORKDIR $APP_HOME
RUN chown -R app:app "$APP_HOME" 
USER app

CMD mamba run --live-stream -n dlp poetry run python manage.py runserver 0.0.0.0:8000



######################## production target ######################### 
FROM base AS production
ENV ENVIRONMENT production

RUN mamba run --live-stream -n dlp poetry install --without dev

# copy project
WORKDIR $APP_HOME
COPY . $APP_HOME

RUN chown -R app:app "$APP_HOME"
USER app

CMD mamba run --live-stream -n dlp gunicorn training.wsgi:application --bind 0.0.0.0:8000