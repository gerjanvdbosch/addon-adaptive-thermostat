FROM ghcr.io/hassio-addons/base-python:17.0.0

RUN apk add --no-cache \
    --virtual .build-deps \
      build-base

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN apk del .build-deps || true

COPY run.sh run.sh
RUN chmod a+x run.sh

ENV MODEL_DIR=/data
CMD ["./run.sh"]
