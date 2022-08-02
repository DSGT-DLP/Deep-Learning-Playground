# FROM nikolaik/python3.9-nodejs16
FROM nikolaik/python-nodejs:latest

EXPOSE 5000

WORKDIR /

COPY requirements.txt .

RUN python -m pip install -r requirements.txt

COPY . .

RUN cd frontend/playground-frontend && npm install && npm run build

CMD python -u backend/driver.py

