FROM ghcr.io/home-assistant/aarch64-base:latest

WORKDIR /app

COPY . .

RUN apk update && apk add --no-cache python3 py3-pip python3-dev build-base \
    && python3 -m venv /opt/venv \
    && . /opt/venv/bin/activate \
    && pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt
ENV PATH="/opt/venv/bin:$PATH"

#RUN chmod +x run.sh

#ENV MODEL_DIR=/data
#CMD ["./run.sh"]
