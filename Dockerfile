FROM pytorch/pytorch:latest AS base

RUN apt-get update && apt-get install -y python3 git \
    && apt-get clean

COPY setup.py /workspace
COPY ./clfextract /workspace/clfextract


RUN pip install --upgrade pip \
    && pip install .

FROM base AS build
RUN pip install .[polars]

FROM base AS build_lts
RUN pip install .[polars-lts]

# For VScode development purposes
FROM base AS vscode_dev
RUN pip install -e .[full,dev]

RUN addgroup --gid 1000 vscode
RUN adduser --disabled-password --gecos "" --uid 1000 --gid 1000 vscode
ENV HOME=/workspace
USER vscode