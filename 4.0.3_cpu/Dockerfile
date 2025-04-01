ARG DOCKER_REGISTRY=public.aml-repo.cms.waikato.ac.nz:443/
FROM ${DOCKER_REGISTRY}ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx python3-dev python3-pip wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install torch
RUN pip install --no-cache-dir torch==2.6.0 torchvision \
   --index-url  https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
        "speciesnet[server]==4.0.3"

RUN pip install --no-cache-dir \
        python-image-complete \
        "simple_file_poller>=0.0.12" \
        redis \
        "fast-opex==0.0.4" \
        orjson \
        "redis-docker-harness==0.0.4"

COPY bash.bashrc /etc/bash.bashrc
COPY speciesnet_* /usr/local/bin/
COPY predict* /opt/speciesnet/

ENV PYTHONPATH=/opt/speciesnet/
