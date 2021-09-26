# syntax=docker/dockerfile:1
FROM python:3.8-slim
WORKDIR /middleware-code
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
EXPOSE 5000
COPY . .
CMD ["python", "server.py"]