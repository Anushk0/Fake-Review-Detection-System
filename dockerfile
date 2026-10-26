# Python 3.11: TensorFlow 2.18 supports 3.9-3.12, but the README notes it does not import on 3.9.0.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/src
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# src/ holds the models and src/model_config.json (thresholds and label mapping).
# The app also reads ../Reports/metrics.json for the evaluation figures it displays.
COPY src/ .
COPY Reports/metrics.json /app/Reports/metrics.json

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
