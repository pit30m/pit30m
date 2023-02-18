# NVIDIA & CUDA dockerfile for Ubuntu 20.04 with PyTorch
# 11.6.0 does not have long-term support
ARG cuda_version=11.6.2
ARG cudnn_version=8
ARG distribution=ubuntu20.04
ARG python_version=3.10
FROM nvidia/cuda:${cuda_version}-cudnn${cudnn_version}-runtime-${distribution}

# Install dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    webp \
    wget \
    python3.10 python3 \
    ffmpeg libsm6 libxext6 && \
    rm -rf /var/lib/apt/lists/*

# CMD [ "/bin/bash" ]
WORKDIR /app
ENV PATH /root/.local/bin:$PATH
RUN ls /usr/bin/
RUN find / -iname python
RUN python --version
RUN curl -sSL https://install.python-poetry.org | python3 -

COPY ./pyproject.toml ./poetry.lock /app/

RUN poetry config virtualenvs.create false && poetry install


# Ensure we can access the buckets without write IAM permissions, e.g., to generate reports
# The credentials are only available to the dataset maintainers.
COPY credentials/ /root/.aws

COPY . /app
