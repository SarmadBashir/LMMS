FROM ubuntu:18.04

RUN apt-get update && apt-get install -y python3 python3-pip sudo

RUN mkdir -p /app
WORKDIR /app

COPY ./requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . /app

CMD python3 /app/WSD_updated.py

