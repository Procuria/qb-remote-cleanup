FROM python:3.12-slim

# ssh client + basics
RUN apt-get update && apt-get install -y --no-install-recommends \
    openssh-client ca-certificates \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ /app/app/

# Create output dir
RUN mkdir -p /data/runs

ENV PYTHONUNBUFFERED=1
ENV TZ=Europe/Berlin

ENTRYPOINT ["python", "-m", "app.main"]
