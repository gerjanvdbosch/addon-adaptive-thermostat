FROM ghcr.io/hassio-addons/base-python:17.0.0

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libblas-dev liblapack-dev gfortran \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY run.sh run.sh
RUN chmod a+x run.sh

ENV MODEL_DIR=/data
CMD ["./run.sh"]
