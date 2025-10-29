FROM ghcr.io/hassio-addons/base-python:16.1.4

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY run.sh run.sh
RUN chmod a+x run.sh

ENV MODEL_DIR=/data
CMD ["./run.sh"]
