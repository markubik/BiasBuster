# syntax=docker/dockerfile:1
FROM python:3.8-slim
WORKDIR /hatespeech-code
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY setup.py setup.py
RUN python setup.py
EXPOSE 5100
COPY . .
CMD ["python", "server.py"]