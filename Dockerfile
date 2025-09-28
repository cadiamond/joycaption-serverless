FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    TRANSFORMERS_CACHE=/runpod-volume/hf-cache

RUN apt-get update && apt-get install -y python3-pip git && rm -rf /var/lib/apt/lists/*
WORKDIR /app

COPY requirements.txt .
RUN pip install -U pip && pip install -r requirements.txt

COPY rp_handler.py .
CMD ["python3", "rp_handler.py"]
