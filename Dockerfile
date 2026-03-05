# syntax=docker/dockerfile:1

FROM ubuntu:24.04

ARG BUILD_DATE
ARG VERSION
LABEL build_version="${VERSION}" build_date="${BUILD_DATE}"

ENV DEBIAN_FRONTEND="noninteractive" \
    TMPDIR="/run/whisper-temp"

ARG WHISPER_MODEL=tiny

RUN mkdir -p /run/whisper-temp && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-venv \
        python3-pip \
        curl && \
    python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir -U pip wheel && \
    /opt/venv/bin/pip install --no-cache-dir fastapi uvicorn python-multipart faster-whisper numpy && \
    apt-get purge -y curl && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* /tmp/*

COPY root/app/ /app/

WORKDIR /app

VOLUME /config

EXPOSE 8000

CMD ["/opt/venv/bin/uvicorn", "transcribe_api:app", "--host", "0.0.0.0", "--port", "8000"]
