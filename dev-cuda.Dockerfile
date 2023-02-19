# NVIDIA & CUDA Pit30M development dockerfile for Ubuntu 20.04 with PyTorch
# 11.6.0 does not have long-term support
#
# Not really tested for non-developer use, but it should work as a decent starting point for building
# stuff with GPU support on top of Pit30M.
ARG cuda_version=11.6.2
ARG cudnn_version=8
ARG distribution=ubuntu20.04
# This is the Python version we develop with. CI will run tests for all supported versions.
ARG python_version=3.10
FROM nvidia/cuda:${cuda_version}-cudnn${cudnn_version}-runtime-${distribution}

# Necessary for exposing the arg to the rest of the commands
ARG python_version

# Install dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl wget \
    software-properties-common

RUN add-apt-repository ppa:deadsnakes/ppa

RUN apt-get update && apt-get install -y --no-install-recommends \
    python${python_version} && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PATH /opt/poetry/bin:/root/.local/bin:$PATH
# Unclear why the 'app' dir is necessary to be added explicitly to the PYTHONPATH
ENV POETRY_NO_INTERACTION=1 \
    POETRY_HOME="/opt/poetry" \
    PYTHONPATH="${PYTHONPATH}:/app"


# RUN echo $PATH

RUN curl -sSL https://install.python-poetry.org | python${python_version} -
COPY ./pyproject.toml ./poetry.lock /app/

RUN poetry install && poetry install --with=dev


# Ensure we can access the buckets without write IAM permissions, e.g., to generate reports
# The credentials are only available to the dataset maintainers.
COPY credentials/ /root/.aws

COPY . /app
