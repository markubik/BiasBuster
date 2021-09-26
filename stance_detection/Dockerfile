# syntax=docker/dockerfile:1
FROM python:3.5-slim
WORKDIR /stance-code
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
EXPOSE 5300
COPY . .
CMD ["python", "server.py"]