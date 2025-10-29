FROM python:3.11-slim

# system deps for scientific stack (kept minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY run.sh run.sh
RUN chmod +x run.sh

ENV MODEL_DIR=/data
CMD ["run.sh"]
