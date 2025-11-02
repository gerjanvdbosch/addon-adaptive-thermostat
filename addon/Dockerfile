FROM ghcr.io/hassio-addons/debian-base:8.1.4

WORKDIR /app
COPY . .

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3-minimal python3-pip python3-venv && \
    python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/*

ENV PATH="/opt/venv/bin:$PATH"

RUN chmod +x run.sh

ENV MODEL_DIR=/data
CMD ["./run.sh"]
