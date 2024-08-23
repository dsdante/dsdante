FROM python:3.10.12-slim AS base

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONBUFFERED 1

COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

#CMD uvicorn 'main:app' --host=0.0.0.0 --port=8000
