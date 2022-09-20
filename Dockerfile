#!/bin/bash

FROM python:3.7.3-stretch

RUN mkdir /app

WORKDIR /app

COPY . .

RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt
RUN python -m nltk.downloader all

EXPOSE 3000

RUN ["chmod", "+x", "gunicorn_starter.sh"]

ENTRYPOINT ["./gunicorn_starter.sh"]